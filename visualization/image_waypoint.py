import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import h5py
import torch
from itertools import islice
from torch.utils.data import DataLoader

from models.image_net import ImagePolicyModel
from dataloader.dataset import SampleData

# === MODEL INITIALIZATION ===
file_path = "E:/marathon.hdf5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

dataset = SampleData(file_path, 1, 5, 5, 1, 5, ["image", "velocity", "command"], ["location"])
dataloader = DataLoader(dataset, 1, shuffle=False)

model = ImagePolicyModel(backbone="resnet34")
model.load_state_dict(torch.load("checkpoints/0627_1556_model.pth", map_location=device, weights_only=True), strict=False)
model.to(device)
model.eval()

# === CONFIGURATION ===
with h5py.File(file_path, 'r') as f:
    print(f"No. of runs: {len(f['runs'])}")
num = input('Enter run no. ')
demo_key = f'{num}'
scale_factor = 2
image_padding = 20
lidar_scale = 7  # Scaling for LiDAR visualization

# === LOAD HDF5 IMAGES AND METADATA ===
with h5py.File(file_path, 'r') as f:
    group = f['runs'][demo_key]['vehicles']['0']
    images = group['image'][:, :, :, :3]
    lasers = group['laser'][:, :, :4]  # top-down view
    velocities = group['velocity'][:]
    locations = group['location'][:]
    controls = group['control'][:]
    commands = group['command'][:]

num_frames = len(dataset) // 1002
height, width = images.shape[1:3]
scaled_width = width * scale_factor
scaled_height = height * scale_factor

data_frames = list(islice(dataloader, int(num_frames*(int(num)-1)), int(num_frames*int(num))))

# === PYGAME SETUP ===
pygame.init()
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

screen_width = 2 * scaled_width + image_padding
screen_height = scaled_height + 150
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Mine View")

# === SLIDER UI SETTINGS ===
slider_x = 50
slider_y = scaled_height + 50
slider_width = 2 * scaled_width + image_padding - 100
slider_height = 10
thumb_radius = 8

# === STATE ===
frame_index = 0
dragging = False
playing = False
playback_fps = 10
min_fps, max_fps = 1, 60

# === PRECOMPUTED LIDAR SURFACES ===
# Assuming: lasers is a list of numpy arrays (each [N, 4]: x, y, z, intensity)
precomputed_lidar_surfaces = []

for frame_lidar in lasers:
    # Create a 3D array for RGB surface, initially dark gray (30, 30, 30)
    surface_array = np.full((scaled_height, scaled_width, 3), 30, dtype=np.uint8)

    # Extract and scale x, y coordinates
    x = frame_lidar[:, 0]
    y = frame_lidar[:, 1]
    intensity = frame_lidar[:, 3]

    px = (scaled_width / 2 + y * lidar_scale).astype(np.int32)
    py = (scaled_height / 2 - x * lidar_scale).astype(np.int32)

    # Filter valid pixel locations
    valid = (px >= 0) & (px < scaled_width) & (py >= 0) & (py < scaled_height)
    px = px[valid]
    py = py[valid]
    intensities = (np.clip(intensity[valid] * 255, 0, 255)).astype(np.uint8)

    # Set pixels: grayscale (R=G=B=intensity)
    surface_array[py, px] = np.stack([intensities]*3, axis=1)

    # Draw ego vehicle
    ego_px = scaled_width // 2
    ego_py = scaled_height // 2
    radius = 5
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                cx = ego_px + dx
                cy = ego_py + dy
                if 0 <= cx < scaled_width and 0 <= cy < scaled_height:
                    surface_array[cy, cx] = [0, 0, 255]

    # Convert numpy array to pygame Surface
    lidar_surface = pygame.surfarray.make_surface(surface_array.swapaxes(0, 1))  # Pygame uses (width, height)
    precomputed_lidar_surfaces.append(lidar_surface)


# === UTILITIES ===
def ego_to_camera(points):
    x, y = points[:, 0], points[:, 1]
    z = np.zeros_like(x)

    # Ego → Camera: [right, down, forward]
    cam = np.stack([y, -z, x], axis=1)
    cam += np.array([0.0, 2.0, 2.0])

    # Apply pitch (-10° down)
    pitch = np.radians(10.0)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    cam = (Rx @ cam.T).T

    # Camera is 2m forward (x), 2m above (z) → offset in camera frame: Z−2, Y−2
    #cam += np.array([0.0, 2.0, 2.0])

    return cam

def project_to_image(cam_pts, image_width=320, image_height=240, fov=90.0):
    fx = fy = image_width / (2 * np.tan(np.radians(fov / 2)))
    cx, cy = image_width / 2, image_height / 2
    x, y, z = cam_pts[:, 0], cam_pts[:, 1], np.clip(cam_pts[:, 2], 1e-5, None)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=1)


# === DRAW FUNCTIONS ===
def draw_info(surface, index, render_fps):
    x_offset = scaled_width + image_padding + 20
    y_start = 20
    line_spacing = 25

    def render(label, array):
        return font.render(f"{label}: {np.round(array[index], 3)}", True, (255, 255, 255))

    info_lines = [
        font.render(f"Playback FPS: {playback_fps} ([ or ])", True, (255, 255, 255)),
        font.render(f"Render FPS: {render_fps:.1f}", True, (255, 255, 255)),
        render("Velocity", velocities),
        render("Position", locations),
        render("Control", controls),
        render("Command", commands),
    ]

    for i, line in enumerate(info_lines):
        surface.blit(line, (x_offset, y_start + i * line_spacing))

def draw_lidar_surface(lidar, actual, predicted):
    canvas = np.full((scaled_height, scaled_width, 3), 30, dtype=np.uint8)
    x, y, intensity = lidar[:, 0], lidar[:, 1], lidar[:, 3]
    px = (scaled_width / 2 + y * lidar_scale).astype(int)
    py = (scaled_height / 2 - x * lidar_scale).astype(int)
    valid = (px >= 0) & (px < scaled_width) & (py >= 0) & (py < scaled_height)
    canvas[py[valid], px[valid]] = np.stack([np.clip(intensity[valid] * 255, 0, 255)] * 3, axis=1)
    def mark(wps, color):
        for wp in wps:
            px = int(scaled_width / 2 + wp[1] * lidar_scale)
            py = int(scaled_height / 2 - wp[0] * lidar_scale)
            if 0 <= px < scaled_width and 0 <= py < scaled_height:
                canvas[py-2:py+2, px-2:px+2] = color
    mark(actual, [0, 255, 0])
    mark(predicted, [255, 0, 0])

    ego_px = scaled_width // 2
    ego_py = scaled_height // 2
    radius = 5
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                cx = ego_px + dx
                cy = ego_py + dy
                if 0 <= cx < scaled_width and 0 <= cy < scaled_height:
                    canvas[cy, cx] = [0, 0, 255]
    
    return pygame.surfarray.make_surface(canvas.swapaxes(0, 1))

def draw_waypoints(surface, waypoints, color):
    img_pts = project_to_image(ego_to_camera(waypoints))
    for pt in img_pts:
        u, v = int(pt[0] * scale_factor), int(pt[1] * scale_factor)
        if 0 <= u < scaled_width and 0 <= v < scaled_height:
            pygame.draw.circle(surface, color, (u, v), 3)

def draw_slider(surface, value):
    pygame.draw.rect(surface, (160, 160, 160), (slider_x, slider_y, slider_width, slider_height))
    pos = slider_x + int((value / max(1, num_frames - 1)) * slider_width)
    pygame.draw.circle(surface, (255, 0, 0), (pos, slider_y + slider_height // 2), thumb_radius)

def get_slider_value(mouse_x):
    relative_x = max(slider_x, min(mouse_x, slider_x + slider_width))
    percent = (relative_x - slider_x) / slider_width
    return int(percent * (num_frames - 1))


# === MAIN LOOP ===
running = True
while running:
    dt = clock.tick(playback_fps if playing else 0)
    render_fps = clock.get_fps()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if slider_y - 10 <= my <= slider_y + slider_height + 10:
                dragging = True
                frame_index = get_slider_value(mx)
                playing = False
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging:
            mx, my = pygame.mouse.get_pos()
            frame_index = get_slider_value(mx)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                playing = not playing
            elif event.key == pygame.K_RIGHT:
                frame_index = min(frame_index + 1, num_frames - 1)
                playing = False
            elif event.key == pygame.K_LEFT:
                frame_index = max(frame_index - 1, 0)
                playing = False
            elif event.key == pygame.K_LEFTBRACKET:
                playback_fps = max(min_fps, playback_fps - 1)
            elif event.key == pygame.K_RIGHTBRACKET:
                playback_fps = min(max_fps, playback_fps + 1)

    if playing:
        frame_index = (frame_index + 1) % num_frames

    frame = images[int(frame_index)][..., ::-1]  # BGR to RGB
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (scaled_width, scaled_height))

    obs, act = data_frames[int(frame_index)]
    image_tensor, vel_tensor, cmd_tensor = [x.to(device) for x in obs]
    target = act[0].to(device)
    with torch.inference_mode():
        pred = model(image_tensor, vel_tensor, cmd_tensor)
    loss = loss_fn(pred, target).item()
    pred_np = pred[0].cpu().numpy()
    target_np = target[0].cpu().numpy()

    draw_waypoints(surface, target_np, (0, 255, 0))
    draw_waypoints(surface, pred_np, (255, 0, 0))

    lidar_surface = draw_lidar_surface(lasers[int(frame_index)], target_np, pred_np)

    screen.fill((255, 255, 255))
    screen.blit(surface, (0, 0))
    screen.blit(lidar_surface, (scaled_width + image_padding, 0))
    label = font.render(f"Frame: {frame_index} | Loss: {loss:.4f} | FPS: {render_fps:.1f}", True, (0, 0, 0))
    screen.blit(label, (10, scaled_height + 10))
    draw_slider(screen, frame_index)
    #draw_info(screen, frame_index, render_fps)
    pygame.display.flip()

pygame.quit()

import pygame
import numpy as np
import h5py

def count_run_with_collisions(file):
    """Count how many runs contain at least one (0, 0, 0) vehicle location."""
    runs = file['runs']
    count = 0
    for run_key in runs:
        locations = runs[run_key]['vehicles']['0']['location'][:]
        if np.any(np.all(locations == [0.0, 0.0, 0.0], axis=1)):
            count += 1
    return count

# === CONFIGURATION ===
file_path = 'D:\\marathon.hdf5'
with h5py.File(file_path, 'r') as f:
    print(f"No. of runs: {len(f['runs'])}")
    print(f"Runs with collisions: {count_run_with_collisions(f)}")
num = input('Enter run no. ')
demo_key = f'{num}'
scale_factor = 3
image_padding = 20
lidar_scale = 2  # Scaling for LiDAR visualization

# === LOAD HDF5 IMAGES AND METADATA ===
with h5py.File(file_path, 'r') as f:
    group = f['runs'][demo_key]['vehicles']['0']
    images = group['image'][:, :, :, :3]
    lasers = group['laser'][:, :, :4]  # top-down view
    velocities = group['velocity'][:]
    accelerations = group['acceleration'][:]
    locations = group['location'][:]
    angular_velocities = group['angular_velocity'][:]
    controls = group['control'][:]
    commands = group['command'][:]
    waypoints = group['waypoint'][:]

num_frames = len(images)
height, width = images.shape[1:3]
scaled_width = width * scale_factor
scaled_height = height * scale_factor

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

    # Draw ego vehicle (red circle)
    ego_px = scaled_width // 2
    ego_py = scaled_height // 2
    radius = 5
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                cx = ego_px + dx
                cy = ego_py + dy
                if 0 <= cx < scaled_width and 0 <= cy < scaled_height:
                    surface_array[cy, cx] = [255, 0, 0]

    # Convert numpy array to pygame Surface
    lidar_surface = pygame.surfarray.make_surface(surface_array.swapaxes(0, 1))  # Pygame uses (width, height)
    precomputed_lidar_surfaces.append(lidar_surface)


# === DRAW FUNCTIONS ===
def draw_slider(surface, value):
    pygame.draw.rect(surface, (160, 160, 160), (slider_x, slider_y, slider_width, slider_height))
    pos = slider_x + int((value / max(1, num_frames - 1)) * slider_width)
    pygame.draw.circle(surface, (255, 0, 0), (pos, slider_y + slider_height // 2), thumb_radius)
    label = font.render(f"Frame: {value}", True, (0, 0, 0))
    surface.blit(label, (10, slider_y - 25))

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
        render("Acceleration", accelerations),
        render("Position", locations),
        render("Angular Velocity", angular_velocities),
        render("Control", controls),
        render("Command", commands),
        render("Waypoint", waypoints),
    ]

    for i, line in enumerate(info_lines):
        surface.blit(line, (x_offset, y_start + i * line_spacing))

def draw_lidar(surface, lidar_surface, origin=(scaled_width + image_padding, 0)):
    surface.blit(lidar_surface, origin)

def get_slider_value(mouse_x):
    relative_x = max(slider_x, min(mouse_x, slider_x + slider_width))
    percent = (relative_x - slider_x) / slider_width
    return int(percent * (num_frames - 1))

# === MAIN LOOP ===
running = True
while running:
    dt = clock.tick(playback_fps if playing else 60)
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

    frame = images[frame_index][..., ::-1]
    surf_agent = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf_agent = pygame.transform.scale(surf_agent, (scaled_width, scaled_height))

    screen.fill((255, 255, 255))
    screen.blit(surf_agent, (0, 0))
    draw_lidar(screen, precomputed_lidar_surfaces[frame_index])
    draw_slider(screen, frame_index)
    draw_info(screen, frame_index, render_fps)
    pygame.display.flip()

pygame.quit()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from models.image_net import ImagePolicyModel
from dataloader.dataset import SampleData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

dataset = SampleData("data/run.hdf5", 1, 5, 5, 1, 5, ["image", "velocity", "command"], ["location"])
dataloader = DataLoader(dataset, 1, shuffle=False)

model = ImagePolicyModel(backbone="resnet34")
model.load_state_dict(torch.load("checkpoints/0627_1556_model.pth", map_location=device, weights_only=True), strict=False)
model.to(device)
model.eval()

obs, act = next(iter(dataloader))
image, velocity, command = [x.to(device) for x in obs]
target = act[0].to(device)

with torch.inference_mode():
    pred = model(image, velocity, command)
    loss = loss_fn(pred, target)

print("Predicted Waypoints:", pred.cpu().numpy())
print("Actual Waypoints:", target.cpu().numpy())
print("Loss:", loss.item())

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from models.image_net import ImagePolicyModel
from models.resnet_utils import OGCoordConverter, CoordConverter
from dataloader.dataset import SampleData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

dataset = SampleData("data/run.hdf5", 1, 5, 0, 1, 5, ["image", "velocity", "command"], ["location"])
dataloader = DataLoader(dataset, 1, shuffle=False)

model = ImagePolicyModel(backbone="resnet34")
model.load_state_dict(torch.load("checkpoints/0624_1540_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

coord_converter = CoordConverter(device=device)

obs, act = next(iter(dataloader))
image, velocity, command = [x.to(device) for x in obs]
target = act[0].to(device)

print(command)

with torch.inference_mode():
    _pred = model(image, velocity, command)
    pred = coord_converter(_pred)
    loss = loss_fn(_pred, target)

print("Predicted Waypoints:", _pred.cpu().numpy())
print("Actual Waypoints:", target.cpu().numpy())
print("Loss:", loss.item())

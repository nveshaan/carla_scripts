import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from models.image_net import ImagePolicyModel
from models.resnet_utils import CoordConverter
from dataloader.dataset import SampleData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

dataset = SampleData("data/sprint.hdf5", 1, 5, 0, 1, 5, ["image", "velocity", "command"], ["location"])
dataloader = DataLoader(dataset, 2, shuffle=False)

model = ImagePolicyModel(backbone="resnet34")
model.load_state_dict(torch.load("checkpoints/0624_0609_epoch1.pth", map_location=device), strict=False)
model.to(device)
model.eval()

coord_converter = CoordConverter()

obs, act = next(iter(dataloader))
image, velocity, command = [x.to(device) for x in obs]
target = act[0].to(device)

print(command)

with torch.inference_mode():
    pred = model(image, velocity, command)
    pred = coord_converter(pred)
    loss = loss_fn(pred, target)

print("Predicted Waypoints:", pred.cpu().numpy())
print("Actual Waypoints:", target.cpu().numpy())
print("Loss:", loss.item())

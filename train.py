from models.image_net import ImagePolicyModel
from dataloader.dataset import SampleData

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader, random_split


def train(dataloader, model, loss_fn, optim):
    model.train()
    size = len(dataloader.dataset)

    for batch, (observations, actions) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(*observations)
        loss = loss_fn(pred, *actions)

        # backpropagation
        loss.backwards()
        optim.step()
        optim.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(observations[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.inference_mode():
        for observations, actions in dataloader:
            pred = model(*observations)
            test_loss += loss_fn(pred, *actions).item()
    
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    print("Starting training...")

    # load data
    dataset = SampleData(
        file_path='D:/marathon.hdf5',
        obs_horizon=1,
        act_horizon=5,
        gap=0,
        obs_freq=1,
        act_freq=1,
        obs_keys=['image', 'velocity', 'command'],
        act_keys=['location']
    )

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    # set hyperparameters
    learning_rate = 1e-3
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImagePolicyModel(backbone='resent34').compile().to(device)
    loss_fn = nn.MSELoss()
    optim = optimizer.Adam(model.parameters(), lr=learning_rate)

    # start training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optim)
        val(val_loader, model, loss_fn)
    print("Done!")
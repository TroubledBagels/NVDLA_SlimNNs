import sys

import torch
import torch.nn as nn
import time

import AlexSNN as AlexModel
from snn_dep.snn_data import load_data

'''
    SNN model class, based off of AlexNet
    Author: Ben Hatton (10903872)
    N.B. AlexNet is not my design, but the implementation (especially the slimmable aspect) is.
'''

def train(model, dataloader, criterion, optimiser, device, wm):
    size = len(dataloader)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()

        pred = model.forward_train(X)
        loss = criterion(pred, y)

        loss.backward()

        optimiser.step()

        if int(batch % (size/10)) == 0:
            loss = loss.item()
            print(f"    loss: {loss:>7f} [{batch:>6d}/{size:>6d}]")

def test(model, dataloader, criterion, device, start_wm):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    model.change_width_mult(start_wm)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model.forward_train(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")

    part = int(sys.argv[1])

    model = AlexModel.AlexSNN(part).to(device)

    print("[OK] Initialising training parameters...")
    criterion = nn.CrossEntropyLoss()

    batch_size = 32

    train_loader, test_loader = load_data("CIFAR10", batch_size, 3)

    wml = model.width_mult_list

    for wm in wml:
        model.change_width_mult(wm)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
        for epoch in range(10):
            print(f"[INFO] Width {wm}, Epoch {epoch + 1}\n-------------------------------")
            start_time = time.time()
            train(model, train_loader, criterion, optimiser, device, wm)
            test(model, test_loader, criterion, device, wm)
            end_time = time.time()
            print(f"[INFO] Time taken: {end_time - start_time:.2f}s")

    torch.save(model, "snn_models/alex_snn.pth")

if __name__ == "__main__":
    main()

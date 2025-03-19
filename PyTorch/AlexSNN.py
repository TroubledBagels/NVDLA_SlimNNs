import sys

import torch
import torch.nn as nn
import time

import snn_dep.slimmable_ops_v2 as slim
from snn_dep.snn_data import load_data

'''
    SNN model class, based off of AlexNet
    Author: Ben Hatton (10903872)
    N.B. AlexNet is not my design, but the implementation (especially the slimmable aspect) is.
'''

class AlexSNN(nn.Module):
    def __init__(self, part: int = 2):
        super(AlexSNN, self).__init__()

        self.width_mult_list = [0.5, 1.0]
        if part == 4:
            self.width_mult_list = [0.25, 0.5, 0.75, 1]
        wml = self.width_mult_list

        self.input_shape = (3, 64, 64)

        print("[OK] Initialising AlexSNN model...")
        self.conv1 = slim.SlimmableConv2d([3, 3, 3, 3], [int(wml[i]*64) for i in range(len(wml))], 11, self.width_mult_list, 4, 2)
        self.MP1 = nn.MaxPool2d(3, 2)
        self.conv2 = slim.SlimmableConv2d([int(wml[i]*64) for i in range(len(wml))], [int(wml[i]*192) for i in range(len(wml))], 5, self.width_mult_list, 1, 2)
        self.MP2 = nn.MaxPool2d(3, 2)
        self.conv3 = slim.SlimmableConv2d([int(wml[i]*192) for i in range(len(wml))], [int(wml[i]*384) for i in range(len(wml))], 3, self.width_mult_list, 1, 1)
        self.conv4 = slim.SlimmableConv2d([int(wml[i]*384) for i in range(len(wml))], [int(wml[i]*256) for i in range(len(wml))], 3, self.width_mult_list, 1, 1)
        self.conv5 = slim.SlimmableConv2d([int(wml[i]*256) for i in range(len(wml))], [int(wml[i]*256) for i in range(len(wml))], 3, self.width_mult_list, 1, 1)
        self.MP3 = nn.MaxPool2d(3, 2)
        # self.AP = nn.AvgPool2d(6, 1)
        self.fc1 = slim.SlimmableLinear([int(wml[i]*256) for i in range(len(wml))], [int(wml[i]*4096) for i in range(len(wml))], self.width_mult_list)
        self.drop = nn.Dropout(0.5)
        self.fc2 = slim.SlimmableLinear([int(wml[i]*4096) for i in range(len(wml))], [int(wml[i]*4096) for i in range(len(wml))], self.width_mult_list)
        self.fc3 = slim.SlimmableLinear([int(wml[i]*4096) for i in range(len(wml))], [10, 10, 10, 10], self.width_mult_list)
        print("[OK] AlexSNN model initialised.")
        print()
        print(self)

    def forward(self, x, confidence_threshold: float = 0.9) -> torch.Tensor:
        width_mult = self.width_mult_list[0]

        while True:
            self.change_width_mult(width_mult)

            x = self.conv1(x)
            x = self.MP1(x)
            x = self.conv2(x)
            x = self.MP2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.MP3(x)
            # x = self.AP(x)
            x = self.drop(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.fc3(x)

            norm_x = torch.norm(x, p=2, dim=0)

            confidence = torch.max(norm_x, 1)[0] - torch.topk(norm_x, 2)[0][:, 1].item()
            if confidence > confidence_threshold:
                break
            elif width_mult == 1.0:
                break
            else:
                width_mult = self.width_mult_list[self.width_mult_list.index(width_mult) + 1]

        return x

    def forward_train(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.MP3(x)
        # x = self.AP(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def change_width_mult(self, wm):
        self.conv1.width_mult = wm
        self.conv2.width_mult = wm
        self.conv3.width_mult = wm
        self.conv4.width_mult = wm
        self.conv5.width_mult = wm
        self.fc1.width_mult = wm
        self.fc2.width_mult = wm
        self.fc3.width_mult = wm

    def __str__(self):
        output = "[INFO] AlexSNN model:\n"
        for name, param in self.named_parameters():
            output += f"    {name} = {param.shape}\n"
        return output

    # def end_training(self):
    #     self.conv1.end_training()
    #     self.conv2.end_training()
    #     self.conv3.end_training()
    #     self.conv4.end_training()
    #     self.conv5.end_training()
    #     self.fc1.end_training()
    #     self.fc2.end_training()
    #     self.fc3.end_training()

def train(model, dataloader, criterion, optimiser, device, wm):
    size = len(dataloader)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()

        # if batch % 100 == 0 or int(batch % (size/10)) == 0:
        #     pred = model(X, wm, True, True)
        # else:
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

    model = AlexSNN(part).to(device)

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
        # model.end_training()

    torch.save(model, "snn_models/alex_snn.pth")

if __name__ == "__main__":
    main()

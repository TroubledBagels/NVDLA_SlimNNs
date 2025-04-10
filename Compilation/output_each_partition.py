from AlexSNN import AlexSNN
import torch
import sys
import torch.nn as nn
import onnx
import os

class StaticAlexSNN(nn.Module):
    def __init__(self, wm):
        super(StaticAlexSNN, self).__init__()

        self.conv1 = nn.Conv2d(3, int(64 * wm), 11, stride=4, padding=2)
        self.MP1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(int(64 * wm), int(192 * wm), 5, padding=2)
        self.MP2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(int(192 * wm), int(384 * wm), 3, padding=1)
        self.conv4 = nn.Conv2d(int(384 * wm), int(256 * wm), 3, padding=1)
        self.conv5 = nn.Conv2d(int(256 * wm), int(256 * wm), 3, padding=1)
        self.MP3 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(int(256 * wm), int(4096 * wm))
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(4096 * wm), int(4096 * wm))
        self.fc3 = nn.Linear(int(4096 * wm), 10)
        print()
        print(self)

    def forward(self, x):
        x = self.conv1(x)
        print(f"[INFO]     Conv1 successful: {x.size()}")
        x = self.MP1(x)
        print(f"[INFO]     MP1 successful: {x.size()}")
        x = self.conv2(x)
        print(f"[INFO]     Conv2 successful: {x.size()}")
        x = self.MP2(x)
        print(f"[INFO]     MP2 successful: {x.size()}")
        x = self.conv3(x)
        print(f"[INFO]     Conv3 successful: {x.size()}")
        x = self.conv4(x)
        print(f"[INFO]     Conv4 successful: {x.size()}")
        x = self.conv5(x)
        print(f"[INFO]     Conv5 successful: {x.size()}")
        x = self.MP3(x)
        print(f"[INFO]     MP3 successful: {x.size()}")
        x = self.drop(x)
        print(f"[INFO]     Drop successful: {x.size()}")
        x = torch.flatten(x, 1)
        print(f"[INFO]     Flatten successful: {x.size()}")
        x = self.fc1(x)
        print(f"[INFO]     FC1 successful: {x.size()}")
        x = self.drop(x)
        print(f"[INFO]     Drop successful: {x.size()}")
        x = self.fc2(x)
        print(f"[INFO]     FC2 successful: {x.size()}")
        x = self.fc3(x)
        print(f"[INFO]     FC3 successful: {x.size()}")

        return x

    def __str__(self):
        output = "[INFO] AlexSNN model:\n"
        for name, param in self.named_parameters():
            output += f"    {name} = {param.shape}\n"
        return output


def main(model_name: str):
    print(f"[OK] Loading model...")
    model = torch.load(f"../PyTorch/snn_models/{model_name}.pth")

    print(f"[OK] Calculating WML...")
    wml = []
    num_parts = 2 if "2_part" in model_name else 4

    if num_parts == 2:
        wml = [0.5, 1.0]
    elif num_parts == 4:
        wml = [0.25, 0.5, 0.75, 1.0]
    else:
        wml = [0.5, 1.0]

    model_list = []
    print(f"[INFO] WML = {wml}")

    for wm in wml:
        print(f"[INFO] Creating model with width multiplier {wm}...")
        model_list.append(StaticAlexSNN(wm))
        print(f"[INFO] Copying weights...")
        model_list[-1].conv1.weight = nn.parameter.Parameter(model.conv1.weight[:int(64 * wm), :3, :, :].cpu())
        model_list[-1].conv1.bias = nn.parameter.Parameter(model.conv1.bias[:int(64 * wm)].cpu())
        model_list[-1].conv2.weight = nn.parameter.Parameter(model.conv2.weight[:int(192 * wm), :int(64 * wm), :, :].cpu())
        model_list[-1].conv2.bias = nn.parameter.Parameter(model.conv2.bias[:int(192 * wm)].cpu())
        model_list[-1].conv3.weight = nn.parameter.Parameter(model.conv3.weight[:int(384 * wm), :int(192 * wm), :, :].cpu())
        model_list[-1].conv3.bias = nn.parameter.Parameter(model.conv3.bias[:int(384 * wm)].cpu())
        model_list[-1].conv4.weight = nn.parameter.Parameter(model.conv4.weight[:int(256 * wm), :int(384 * wm), :, :].cpu())
        model_list[-1].conv4.bias = nn.parameter.Parameter(model.conv4.bias[:int(256 * wm)].cpu())
        model_list[-1].conv5.weight = nn.parameter.Parameter(model.conv5.weight[:int(256 * wm), :int(256 * wm), :, :].cpu())
        model_list[-1].conv5.bias = nn.parameter.Parameter(model.conv5.bias[:int(256 * wm)].cpu())
        model_list[-1].fc1.weight = nn.parameter.Parameter(model.fc1.weight[:int(4096 * wm), :int(256 * wm)].cpu())
        model_list[-1].fc1.bias = nn.parameter.Parameter(model.fc1.bias[:int(4096 * wm)].cpu())
        model_list[-1].fc2.weight = nn.parameter.Parameter(model.fc2.weight[:int(4096 * wm), :int(4096 * wm)].cpu())
        model_list[-1].fc2.bias = nn.parameter.Parameter(model.fc2.bias[:int(4096 * wm)].cpu())
        model_list[-1].fc3.weight = nn.parameter.Parameter(model.fc3.weight[:10, :int(4096 * wm)].cpu())
        model_list[-1].fc3.bias = nn.parameter.Parameter(model.fc3.bias[:10].cpu())
        print(f"[OK] Model created with width multiplier {wm}")

    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"[OK] Saving models...")
    for i, m in enumerate(model_list):
        print(f"[INFO] Saving models for width multiplier {wml[i]}...")
        torch.save(m, f"./partitioned_networks/pth/{model_name}_part_{i + 1}.pth")
        torch.onnx.export(m, dummy_input, f"./partitioned_networks/onnx/{model_name}_part_{i + 1}.onnx", opset_version=8)
        checking_model = onnx.load(f"./partitioned_networks/onnx/{model_name}_part_{i + 1}.onnx")
        onnx.checker.check_model(checking_model)
    print(f"[OK] Models saved.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 output_each_partition.py <model_name>")
        print("List models with -l")
        sys.exit(1)
    if "-l" in sys.argv:
        print("[INFO] Available models:")
        # iterate through ../PyTorch/snn_models
        for filename in os.listdir("../PyTorch/snn_models"):
            if filename.endswith(".pth"):
                print(f"    {filename[:-4]}")
        exit(0)
    main(sys.argv[1])

import torch
import torch.nn as nn

import snn_dep.slimmable_ops_v2 as slim


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

    def forward(self, x: torch.Tensor, confidence_threshold: float = 0.9) -> torch.Tensor:
        width_mult = self.width_mult_list[0]
        original = x.clone()

        while True:
            self.change_width_mult(width_mult)
            x = original
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

            softmax_x = nn.Softmax(dim=1)(x)

            confidence = torch.max(softmax_x, 1)[0] - torch.topk(softmax_x, 2)[0][:, 1].item()
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
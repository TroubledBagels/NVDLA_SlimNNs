import torch.nn as nn

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, wml):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(wml)
        self.width_mult_list = wml
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list, kernel_size, wml, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(max(in_channels_list), max(out_channels_list), kernel_size, stride=stride,
                                              padding=padding, dilation=dilation, groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult_list = wml
        self.width_mult = max(self.width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, wml, bias=True):
        super(SlimmableLinear, self).__init__(max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult_list = wml
        self.width_mult = max(self.width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = nn.functional.linear(input, weight, bias)
        return y

def make_divisible(v, divisor=8, min_value=1):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]

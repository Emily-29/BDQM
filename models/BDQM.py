import math
import torch
import torch.nn as nn


class CA(nn.Module):
    def __init__(self, channel_in):
        super(CA, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=channel_in, out_features=channel_in), nn.Sigmoid())

    def forward(self, x):
        out = self.gap(x)
        out = out.view(out.size(0), -1)
        out = ((self.fc(out)).unsqueeze(2)).unsqueeze(3)
        out = torch.mul(x, out)
        return out


class MA(nn.Module):
    def __init__(self, channel_in):
        super(MA, self).__init__()

        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=(1, 3), padding=(0, 1)), nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=(1, 3), padding=(0, 1)))

        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=(3, 1), padding=(1, 0)), nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=(3, 1), padding=(1, 0)))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.add(self.conv_w(x), self.conv_h(x))
        out = torch.mul(x, self.sigmoid(out))
        return out


class PA(nn.Module):
    def __init__(self, patch_nums):
        super(PA, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=patch_nums, out_channels=patch_nums, kernel_size=(1, 1), groups=patch_nums), nn.Sigmoid())

    def forward(self, x):
        out = torch.mul(self.conv(x), x)
        return out


class MultiConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(MultiConv, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=(1, 3), padding=(0, 1)), nn.LeakyReLU())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU())
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=(3, 1), padding=(1, 0)), nn.LeakyReLU())

    def forward(self, x):
        out = torch.add(torch.add(self.conv_1(x), self.conv_2(x)), self.conv_3(x))
        return out


class PIE(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(PIE, self).__init__()

        self.conv = nn.Sequential(
            CA(channel_in=channel_in),
            MultiConv(channel_in=channel_in, channel_out=channel_out),
            MultiConv(channel_in=channel_out, channel_out=channel_out))
        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_channels=channel_in + channel_out, out_channels=channel_out, kernel_size=(1, 1)), nn.LeakyReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        out0 = self.conv_1x1(torch.cat((self.conv(x), x), dim=1))
        out = self.maxpool(out0)
        return out, out0


class IRNet(nn.Module):
    def __init__(self, patch_nums, drop_rate):
        super(IRNet, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.pa = PA(patch_nums=patch_nums)
        self.ma = MA(channel_in=patch_nums)

        self.conv_iqa = nn.Sequential(
            nn.Conv2d(in_channels=patch_nums, out_channels=patch_nums, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(),
            nn.Conv2d(in_channels=patch_nums, out_channels=2 * patch_nums, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * patch_nums, out_features=patch_nums), nn.LeakyReLU(), nn.Dropout(p=drop_rate),
            nn.Linear(in_features=patch_nums, out_features=1))

    def forward(self, x):
        x = self.gap(x)
        x = (x.squeeze(2)).squeeze(2)
        new_size = int(math.sqrt(x.shape[1]))
        x = (x.view(x.size(0), new_size, new_size)).unsqueeze(0)

        x = self.pa(x)
        x = self.ma(x)

        out = self.conv_iqa(x)
        out = (out.squeeze(2)).squeeze(2)
        out = self.fc(out)
        return out, x


class MyNet(nn.Module):
    def __init__(self, cfg):
        super(MyNet, self).__init__()

        self.drop_rate = cfg.drop_rate
        self.patch_nums = cfg.test_dataset.patch_num

        self.LFE = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.PIE_1 = PIE(channel_in=32, channel_out=64)
        self.PIE_2 = PIE(channel_in=64, channel_out=128)
        self.PIE_3 = PIE(channel_in=128, channel_out=256)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.irnet = IRNet(patch_nums=self.patch_nums, drop_rate=self.drop_rate)

    def forward_hf(self, x):
        out = self.LFE(x)

        out, out0 = self.PIE_1(out)
        out, _ = self.PIE_2(out)
        out, _ = self.PIE_3(out)

        out = self.conv(out)
        return out, out0

    def forward_ir(self, x):
        out, extra = self.irnet(x)
        return out, extra

    def forward(self, x):
        f, extra = self.forward_hf(x)
        score, _ = self.forward_ir(f)
        return score, extra

import torch
from torch import nn
import torch.nn.functional as F


class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, stride=1,padding=1,dilation=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class channel_wise(nn.Module):
    def __init__(self):
        super(channel_wise, self).__init__()
        self.softmax   = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B,C,H,W = x.size()
        x_re = x.reshape(B,C,-1)
        x_tp = x_re.transpose(1,2)
        channel_wise = torch.matmul(x_re,x_tp)
        channel_wise = self.softmax(channel_wise)
        x_channel = torch.matmul(channel_wise,x_re)
        x_channel = x_channel.reshape(B,C,H,W)
        return self.gamma*x_channel

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sigmoid=nn.Sigmoid()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(3,3),stride=1,padding=1)
        self.bn=nn.BatchNorm2d(1)
    def forward(self,x):
        g=torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        g=self.conv(g)
        g=self.bn(g)
        g=self.sigmoid(g)
        return g*x


class shift_channel(nn.Module):
    def __init__(self,in_channels,r):
        super(shift_channel, self).__init__()
        self.fc1 = nn.Conv2d(in_channels*4, in_channels // r, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x0 = x[:, :,0::2, 0::2]
        x1 = x[:, :,1::2, 0::2]
        x2 = x[:, :,0::2, 1::2]
        x3 = x[:, :,1::2, 1::2]
        y = torch.cat([x0, x1, x2, x3], 1)
        y=y.mean((2,3),keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        x = self.sigmoid(y) * x

        return x

class shift_channel_one(nn.Module):
    def __init__(self,in_channels,r):
        super(shift_channel_one, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // r, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // r, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        y = x.mean((2,3),keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        x = self.sigmoid(y) * x

        return x



class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels,r,nums=3):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dense_block = Single_level_densenet(out_channels,nums)
        self.c_att = shift_channel(out_channels,r)


    def forward(self, x):

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.dense_block(x1)
        y = self.c_att(x2)
        x = F.max_pool2d(y, 2, stride=2)

        return x, y



class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels,r):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x_cat = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x_cat)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MADNet(nn.Module):
    def __init__(self, args):
        in_chan = 3
        out_chan = 2
        super(MADNet, self).__init__()
        self.down1 = Downsample_block(in_chan, 32,r=4)
        self.down2 = Downsample_block(32, 64,r=8)
        self.down3 = Downsample_block(64, 128,r=16)
        self.down4 = Downsample_block(128, 256,r=32)
        self.conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.up4 = Upsample_block(512, 256,r=32)
        self.up3 = Upsample_block(256, 128,r=16)
        self.up2 = Upsample_block(128, 64,r=8)
        self.up1 = Upsample_block(64, 32,r=4)
        self.outconv = nn.Conv2d(32, out_chan, 1)
        self.c_att = channel_wise()
        self.s_att = SpatialAttention()


    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.c_att(x)+x+self.s_att(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x4 = self.up4(x, y4)
        x3 = self.up3(x4, y3)
        x2 = self.up2(x3, y2)
        x1 = self.up1(x2, y1)
        out = self.outconv(x1)

        return out


if __name__ == '__main__':
    print('')
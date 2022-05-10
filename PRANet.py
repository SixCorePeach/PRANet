import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from Models.networks.channel_attention_layer import SE_Conv_Block as SE_Conv_Block
from Models.networks.channel_attention_layer import SE_Conv_Block2 as SE_Conv_Block2
from Models.networks.channel_attention_layer import SE_Conv_Block3 as SE_Conv_Block3


class basedModule_1(nn.Module):

    def __init__(self, in_channels, out_channel):
        super(basedModule_1, self).__init__()
        self.mode_down1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=3, padding=2),
            nn.GroupNorm(4, out_channel, eps=1e-5, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(0.5))

    def forward(self, x):
        out_1 = self.mode_down1(x)
        return out_1

class basedModule_4(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(basedModule_4, self).__init__()
        self.mode_up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channel, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        out_1 = self.mode_up1(x)
        return out_1


class basedModule_5_1(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(basedModule_5_1, self).__init__()

        self.mode_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channel, eps=1e-5, affine=True),
            nn.ReLU())

    def forward(self, x):
        out_1 = self.mode_up1(x)
        return out_1


class DoubleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DoubleConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        bn_out_1 = self.bn_1(conv_out_1)
        relu_out = F.relu(bn_out_1)
        conv_out_2 = self.conv_2(relu_out)
        bn_out_2 = self.bn_2(conv_out_2)
        return F.relu(bn_out_2)


class basedModule_5(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(basedModule_5, self).__init__()

        self.mode_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout2d(0.5))

    def forward(self, x):
        out_1 = self.mode_up1(x)
        return out_1


class basedModule_9(nn.Module):

    def __init__(self, in_channels, out_channel):
        super(basedModule_9, self).__init__()
        self.mode_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mode_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mode_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mode_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.mode_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.mode_max(x)
        out_1 = self.mode_conv1(x)
        out_2 = self.mode_conv2(x)
        out_3 = self.mode_conv3(x)
        out_4 = self.mode_conv4(x)
        out_end = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        return out_end


class UpSamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
        super(UpSamplingBlock, self).__init__()
        self.tran_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        tran_conv_out = self.tran_conv(x)
        return tran_conv_out


class TTModule(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super().__init__()

        self.conv_0_1 = basedModule_5(64, 64)   # one conv 3*3
        self.conv_1 = basedModule_5(224, 64)   # one conv 3*3

        self.conv0 = basedModule_5(32, 64)
        self.conv5 = basedModule_5(736, 64)

        self.conv9 = basedModule_5_1(128, 64)

        self.up1 = SE_Conv_Block2(128, 64)

        self.up2 = SE_Conv_Block2(448, 224)

        self.conv_1_1 = basedModule_9(32, 32)
        self.conv_1_2 = basedModule_9(64, 64)

        self.conv_4_1 = basedModule_4(128, 128)
        self.conv_4_2 = basedModule_4(448, 448)




    def forward(self, x):

        down1 = self.conv_1_1(x)  # 128

        step_0 = self.conv0(x)    # 32 - 64

        channel_1, atten_1 = self.up1(down1)  #128

        down1_2 = self.conv_4_1(channel_1)  # 128

        down1_1 = self.conv9(down1)  # 128 - 64

        step_1 = torch.cat([step_0, down1_2, x], dim=1)  # 64+32+128=224

        center_2 = self.conv_1(step_1)  # 64

        down2 = self.conv_1_2(center_2)  # 64 - 256

        down2_end = torch.cat([down1, down1_1, down2], dim=1)  # 128 + 64 + 256 = 448

        channel_2, atten_2 = self.up2(down2_end) # 448

        down2_2 = self.conv_4_2(channel_2)  # 448 - 448

        center_2_end = self.conv_0_1(center_2)  # 64 - 64

        step_2 = torch.cat([center_2_end, down2_2, step_1], dim=1)  # 64 + 448 + 224 = 736
        center_6 = self.conv5(step_2)  # 736 - 64

        return down1, center_6

class TNNet_light_BG(nn.Module):

    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super().__init__()
        self.conv_0 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1), nn.Softmax2d())
        self.conv_0_1 = basedModule_5(64, 64)   # one conv 3*3
        self.conv_1 = basedModule_5(64, 64)   # one conv 3*3

        self.conv0 = basedModule_5(3, 29)
        self.conv5 = basedModule_5(288, 64)

        self.conv9 = basedModule_5_1(32, 64)

        self.up1 = SE_Conv_Block(32, 16)

        self.up2 = SE_Conv_Block(160, 80)

        self.conv_1_1 = basedModule_9(3, 8)
        self.conv_1_2 = basedModule_9(64, 16)

        self.conv_4_1 = basedModule_4(32, 32)
        self.conv_4_2 = basedModule_4(160, 160)




    def forward(self, x):

        down1 = self.conv_1_1(x)  # 3 - 32

        step_0 = self.conv0(x)    # 3 - 29

        channel_1, atten_1 = self.up1(down1)  #32-32

        down1_2 = self.conv_4_1(channel_1)  # 32 - 32

        down1_1 = self.conv9(down1)  # 32 - 64

        step_1 = torch.cat([step_0, down1_2, x], dim=1)  # 32+3+29=64

        center_2 = self.conv_1(step_1)  # 64 - 64

        down2 = self.conv_1_2(center_2)  # 64 - 64

        down2_end = torch.cat([down1, down1_1, down2], dim=1)  # 32 + 64 + 64 =160

        channel_2, atten_2 = self.up2(down2_end) # 160

        down2_2 = self.conv_4_2(channel_2)  # 160 - 160

        center_2_end = self.conv_0_1(center_2)  # 64 - 64

        step_2 = torch.cat([center_2_end, down2_2, step_1], dim=1)  # 64 + 160 + 64 = 288

        center_6 = self.conv5(step_2)  # 288 - 32
        return down1, center_6

class TNNet_lighter_BG(nn.Module):

    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super().__init__()

        self.conv0 = basedModule_5(128, 128)
        self.conv5 = basedModule_5(768, 384)
        self.up1 = SE_Conv_Block3(512, 256)
        self.conv_1_1 = basedModule_9(128, 128)
        self.conv_4_1 = basedModule_4(512, 512)

    def forward(self, x):

        down1 = self.conv_1_1(x)  # 512
        step_0 = self.conv0(x)    # 128
        channel_1, atten_1 = self.up1(down1)  #512
        down1_2 = self.conv_4_1(channel_1)  # 512
        step_1 = torch.cat([step_0, down1_2, x], dim=1)  # 512+256 = 768
        center_6 = self.conv5(step_1)  # 768 - 32
        return center_6

class PRANet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super().__init__()
        self.TTmodel1 = TNNet_light_BG(args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1))
        self.TTmodel2 = TTModule(args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1))
        self.TTmodel3 =TNNet_lighter_BG(args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1))

        self.down_sampling_1 = nn.MaxPool2d(2, 2)

        self.down_sampling_2 = nn.MaxPool2d(2, 2)

        self.conv_block_1 = DoubleConvBlock(384, 192)

        self.conv_block_2 = DoubleConvBlock(256, 128)

        self.conv_block_3 = DoubleConvBlock(32, 64)

        self.up_sampling_1 = UpSamplingBlock(384, 384)

        self.up_sampling_2 = UpSamplingBlock(192, 192)

        self.up_sampling_3 = UpSamplingBlock(256, 256)

        self.up_sampling_4 = UpSamplingBlock(128, 128)

        self.conv_0 = nn.Sequential(nn.Conv2d(192, 2, kernel_size=1), nn.Softmax2d())

    def forward(self, x):
        downsampling_1, out_model1 = self.TTmodel1(x)
        inputs_model2 = self.down_sampling_1(downsampling_1)

        downsampling_2, out_model2 = self.TTmodel2(inputs_model2)

        inputs_model3 = self.down_sampling_1(downsampling_2)

        out1 = self.TTmodel3(inputs_model3) # 384
        upsample2 = self.up_sampling_1(out1)
        upsample3 = self.conv_block_1(upsample2)
        upsample3 = self.up_sampling_2(upsample3)# 192

        out2 = torch.cat([out_model2, upsample3], dim=1) # 256
        upsample4 = self.up_sampling_3(out2) #256
        upsample5 = self.conv_block_2(upsample4)
        upsample5 = self.up_sampling_4(upsample5)#128

        out3 = torch.cat([out_model1, upsample5], dim=1)#160

        end = self.conv_0(out3)
        return end


if __name__ == '__main__':
    temp = torch.randn((1, 3, 256, 320))
    net = TNNet_light_BG()
    s = net(temp)
    print(s, s.shape)

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2f" % (total))



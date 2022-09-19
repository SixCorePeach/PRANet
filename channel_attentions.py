# # SE block add to U-net

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


class SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out
                                                                 #(384, 384)
        self.globalAvgPool = nn.AvgPool2d((192, 192), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((192, 192), stride=1)  # (128, 150) for ISIC2017
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1

        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


class SE_Conv_Block3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((12, 12), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((12, 12), stride=1)  # (128, 150) for ISIC2017
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1

        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


class SE_Conv_Block2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((48, 48), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((48, 48), stride=1)  # (128, 150) for ISIC2017
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1

        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


class SE_Conv_Block_mogai(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_mogai, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((150, 112), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((150, 112), stride=1)  # (128, 150) for ISIC2017
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out1 = self.globalMaxPool(out1)

        out = out.view(out.size(0), -1)
        out1 = out1.view(out1.size(0), -1)
        out_in = out + out1
        out_in = self.fc1(out_in)
        out_in = self.relu(out_in)
        out_in = self.fc2(out_in)
        out_in = self.sigmoid(out_in)
        out_in = out.view(out_in.size(0), out_in.size(1), 1, 1)

        out = out_in * original_out

        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)
        return out, out_in


class SE_Conv_Block_17(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_17, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((150, 128), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((150, 128), stride=1)  # (128, 150) for ISIC2017
        # self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes))
        # self.fc2 = nn.Linear(in_features=round(planes), out_features=planes * 2)
        self.fc1 = nn.Linear(in_features=inplanes, out_features=round(planes) // 2)
        self.fc2 = nn.Linear(in_features=round(planes) // 2, out_features=inplanes)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        # print('^^^^^^^^^^^^^^^', out.shape)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


class SE_Conv_Block_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.globalAvgPool = nn.AvgPool2d((150, 112), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((150, 112), stride=1)
        # self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        # self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes))
        self.fc2 = nn.Linear(in_features=round(planes), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2), )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = torch.mul(out, out)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out = torch.mul(out, out)
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


class SE_Conv_Block_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes * 2)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.IF = nn.Parameter(torch.rand(2 * planes))

        self.globalAvgPool = nn.AvgPool2d((150, 112), stride=1)  # (112, 150) for ISIC2018
        self.globalMaxPool = nn.MaxPool2d((150, 112), stride=1)
        # self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        # self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes))
        self.fc2 = nn.Linear(in_features=round(planes), out_features=planes * 2)

        self.sigmoid = nn.Sigmoid()
        """
        self.downchannel = None
        if inplanes != planes:
        self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * 2),)
        """

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        """
        if self.downchannel is not None:
            residual = self.downchannel(x)
        """
        original_out = out
        out1 = out

        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        # print(out.shape,'######')
        out = self.fc1(out)
        # print(out.shape, '$$$$$')
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = torch.mul(out, out)

        out = self.IF * out
        # print('---------------------', out.shape)


        avg_att = out

        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = torch.mul(out1, out1)
        out1 = (1 - self.IF) * out1
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)

        max_att = out1

        att_weight = max_att + avg_att
        # print(self.IF.shape, att_weight.shape)
        out = original_out * att_weight
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight

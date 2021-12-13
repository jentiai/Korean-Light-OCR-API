import torch.nn as nn
import torch.nn.functional as F

# Add MobileNetV3 from Pytorch official
import torchvision
from torchvision.models import mobilenetv2, mobilenetv3
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

class MobileV3_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel, stride = 2, t = 'small'):
        super(MobileV3_FeatureExtractor, self).__init__()
        # output_channel: dummy variable which does not used
        # t: ['small', 'large']
        if t == 'small':
            backbone = mobilenet_v3_small(pretrained = False)
        else:
            backbone = mobilenet_v3_large(pretrained = False)
        
        # modify backbone
        invert_count = 0
        for name, module in backbone.named_modules():
            if isinstance(module, mobilenetv3.InvertedResidual):
                invert_count += 1
                if invert_count in [1, 2, 4, 9]:
                    conv_count = 0

                    for j in range(len(module.block)):
                        if isinstance(module.block[j], mobilenetv2.ConvBNActivation):
                            conv_count += 1
                            if invert_count == 1:
                                if conv_count == 1:
                                    module.block[j][0] = nn.Conv2d(16, 16, kernel_size = (3,3), stride = (stride,1), padding = (1,1), groups = 16, bias = False)
                            elif conv_count == 2:
                                if invert_count == 2:
                                    module.block[j][0] = nn.Conv2d(72, 72, kernel_size = (3,3), stride = (2,1), padding = (1,1), groups = 72, bias = False)    
                                elif invert_count == 4:
                                    module.block[j][0] = nn.Conv2d(96, 96, kernel_size = (5,5), stride = (2,1), padding = (2,2), groups = 96, bias = False)
                                elif invert_count == 9:
                                    module.block[j][0] = nn.Conv2d(288, 288, kernel_size = (5,5), stride = (2,1), padding = (2,2), groups = 288, bias = False)
        modules = list(backbone.children())[:-2] # -1 for classifer / -2 for Avgpool2d
        self.backbone = nn.Sequential(*modules)
        self.backbone[0][0][0] = nn.Conv2d(input_channel, 16, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False)
        self.backbone[-1][-1][0] = nn.Conv2d(96, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone[-1][-1][1] = nn.BatchNorm2d(output_channel, eps=1e-3)
                                
    def forward(self, x):
        res = self.backbone(x)
        return res

class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """
    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x

if __name__ == '__main__':    
    feature_extractor = MobileV3_FeatureExtractor(3, 576)
    print(feature_extractor)
    import numpy as np
    from torchinfo import summary
    
    summary(feature_extractor, (1, 3, 32, 100))
    
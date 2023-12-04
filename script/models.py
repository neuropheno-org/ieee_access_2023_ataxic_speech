#Pytorch
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional



class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`." # from pytorch
    def __init__(self, sz:Optional[int]=None): 
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.dropout1 = nn.Dropout(0.15)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.dropout2 = nn.Dropout(0.15)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.dropout1 = nn.Dropout(0.15)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.dropout2 = nn.Dropout(0.15)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.dropout3 = nn.Dropout(0.15)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class FC_Resnet_(ResNet):
    def __init__(self, num_layers = 1, num_classes=1000, pretrained=False, model_size = '18', **kwargs):

        # Start with standard resnet defined here
        if model_size == '18':
            super().__init__(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        elif model_size == '34':
            super().__init__(block = BasicBlock, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50':
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101':
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        elif model_size == '152':
            super().__init__(block = Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        else:
            assert print("Error: not accepted size. Must be one of '18','34','50','101','152','50_2','101_2' ")
        #if pretrained:
        #    state_dict = torch.hub.load_state_dict_from_url( models.resnet.model_urls["resnet18"], progress=True)
        #    self.load_state_dict(state_dict)

        self.num_classes = num_classes
        
        #    adjust the first conv layer to accpet monochrome input
        self.conv1 = nn.Conv2d(num_layers, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Convert the original fc layer to a convolutional layer.
        #self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        #self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        #self.last_conv.bias.data.copy_ (self.fc.bias.data)

        self.dropout = nn.Dropout(0.2)

    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.view([-1,self.num_classes])
        return x
    
    
    
class FullyConvolutionalResnet_(ResNet):
    def __init__(self, num_layers = 1, num_classes=1000, pretrained=False, model_size = '18', **kwargs):

        # Start with standard resnet defined here
        if model_size == '18':
            super().__init__(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        elif model_size == '34':
            super().__init__(block = BasicBlock, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50':
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101':
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        elif model_size == '152':
            super().__init__(block = Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        else:
            assert print("Error: not accepted size. Must be one of '18','34','50','101','152','50_2','101_2' ")
        #if pretrained:
        #    state_dict = torch.hub.load_state_dict_from_url( models.resnet.model_urls["resnet18"], progress=True)
        #    self.load_state_dict(state_dict)

        self.num_classes = num_classes
        
        #    adjust the first conv layer to accpet monochrome input
        self.conv1 = nn.Conv2d(num_layers, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        #self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        #self.last_conv.bias.data.copy_ (self.fc.bias.data)

        self.dropout = nn.Dropout(0.2)

    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.last_conv(x)

        x = x.view([-1,self.num_classes])
        return x

    

class FullyConvolutionalResnet(ResNet):
    def __init__(self, num_layers = 1, num_classes=1000, pretrained=False, model_size = '18', **kwargs):

        # Start with standard resnet defined here
        if model_size == '18':
            super().__init__(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        elif model_size == '34':
            super().__init__(block = BasicBlock, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50':
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101':
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        elif model_size == '152':
            super().__init__(block = Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        else:
            assert print("Error: not accepted size. Must be one of '18','34','50','101','152','50_2','101_2' ")
        
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url( models.resnet.model_urls["resnet34"], progress=True)
            self.load_state_dict(state_dict)

        self.num_classes = num_classes
        
        #    adjust the first conv layer to accpet monochrome input
        self.conv1 = nn.Conv2d(num_layers, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Convert the original fc layer to a convolutional layer.
        #self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        #self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        #self.last_conv.bias.data.copy_ (self.fc.bias.data)

        self.dropout = nn.Dropout(0.2)
        
        self.fc_1 = nn.Linear(1024 , 512, bias=True)
        self.fc_2 = nn.Linear(512 , 64, bias=True)
        self.fc_3 = nn.Linear(64 , 1, bias=True)
        
        self.BN1024 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.BN512 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.avgpool = AdaptiveConcatPool2d()
        
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.BN1024(x)
        x = self.dropout(x)
        
        x= self.leakyrelu(self.fc_1(x))
        x = self.BN512(x)
        x = self.dropout(x)
        
        x= self.leakyrelu(self.fc_2(x))
        
        x= self.fc_3(x)

        x = x.view([-1,self.num_classes])
        return x

    
    

#Defining an FCNN
class Speech_model_classification_CNN(torch.nn.Module):
    def __init__ (self, n_classes):
        super(Speech_model_classification_CNN,self).__init__()
        self.n_classes = n_classes

        self.C1 = nn.Conv2d(1,32,5,padding=1)
        self.C2 = nn.Conv2d(32,32,5,padding=1)
        self.C3 = nn.Conv2d(32,64,5,padding=1)
        self.C4 = nn.Conv2d(64,64,5,padding=1)

        self.BN32 = nn.BatchNorm2d(32)
        self.BN64 = nn.BatchNorm2d(64)
        self.BN_class = nn.BatchNorm2d(self.n_classes)
        self.maxpool1 = nn.MaxPool2d(2,2)


        self.fc_cv1 =  nn.Conv2d(64,64,1,padding=0)
        self.fc_cv2 = nn.Conv2d(64,self.n_classes,1,padding=0)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):

        x = self.dropout(F.relu(self.BN32(self.C1(x))))
        x = self.maxpool1(F.relu(self.BN32(self.C2(x))))
        x = self.dropout(F.relu(self.BN64(self.C3(x))))
        x = self.maxpool1(F.relu(self.BN64(self.C4(x))))
        x = self.dropout(F.relu(self.BN64(self.C4(x))))
        x = self.maxpool1(F.relu(self.BN64(self.C4(x))))
        x = self.dropout(F.relu(self.BN64(self.C4(x))))
        x = self.dropout(F.relu(self.BN64(self.C4(x))))

        x = F.relu(self.BN64(self.dropout(self.fc_cv1(x))))

        x = self.BN_class(self.fc_cv2(x))

        x = self.avgpool(x)

        x = x.view([-1,self.n_classes])

        return x
    
#Defining an FCNN
class FCNN_short(torch.nn.Module):
    def __init__ (self, n_classes):
        super(FCNN_short,self).__init__()
        self.n_classes = n_classes

        #self.C1 = nn.Conv2d(1,100,7,padding=1)
        self.C1 = nn.Conv2d(4,100,7,padding=1)
        self.C2 = nn.Conv2d(100,150,5,padding=1)
        self.C3 = nn.Conv2d(150,200,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.maxpool2 = nn.MaxPool2d((12,4),1)


        self.fc_cv1 =  nn.Conv2d(200,128,1,padding=0)
        self.fc_cv2 = nn.Conv2d(128,self.n_classes,1,padding=0)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        
        x = self.C1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C3(x)
        x = F.relu(x)
        x =self.maxpool2(x)

        x = F.relu(self.dropout(self.fc_cv1(x)))
        x = self.fc_cv2(x)
        x = self.avgpool(x)

        x = x.view([-1,self.n_classes])

        return x

    
#Defining an CNN
class CNN_short_fc_wide(torch.nn.Module):
    def __init__ (self, n_classes, n_channels = 1):
        super(CNN_short_fc_wide,self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.C1 = nn.Conv2d(self.n_channels,50,7,padding=1)
        self.C2 = nn.Conv2d(50,100,5,padding=1)
        self.C3 = nn.Conv2d(100,200,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,3)
        self.maxpool2 = nn.MaxPool2d((3,3),1)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc_cv1 =  nn.Linear(12800,128)#nn.Linear(200,128)46400
        self.fc_cv2 = nn.Linear(128,self.n_classes)
        self.dropout = nn.Dropout(0.10)
        self.dropout_cv = nn.Dropout(0.10)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.C1(x)
        x = F.relu(self.dropout_cv(x))
        x = self.maxpool1(x)
        x = self.C2(x)
        x = F.relu(self.dropout_cv(x))
        x = self.maxpool1(x)
        x = self.C3(x)
        x = F.relu(self.dropout_cv(x))
        x =self.maxpool2(x)
        
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout(self.fc_cv1(x)))
        x = self.fc_cv2(x)

        x = x.view([-1,self.n_classes])
        return x    


#Defining an CNN
class CNN_short_fc(torch.nn.Module):
    def __init__ (self, n_classes, n_channels = 1):
        super(CNN_short_fc,self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.C1 = nn.Conv2d(self.n_channels,50,7,padding=1)
        self.C2 = nn.Conv2d(50,100,5,padding=1)
        self.C3 = nn.Conv2d(100,200,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.maxpool2 = nn.MaxPool2d((8,4),1)


        self.fc_cv1 =  nn.Linear(200,128)
        self.fc_cv2 = nn.Linear(128,self.n_classes)
        self.dropout = nn.Dropout(0.5)
        self.dropout_cv = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        
        x = self.C1(x)
        x = F.relu(self.dropout_cv(x))
        x = self.maxpool1(x)
        x = self.C2(x)
        x = F.relu(self.dropout_cv(x))
        x = self.maxpool1(x)
        x = self.C3(x)
        x = F.relu(self.dropout_cv(x))
        x =self.maxpool2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout(self.fc_cv1(x)))
        x = self.fc_cv2(x)

        x = x.view([-1,self.n_classes])

        return x    

#Defining an FCNN
class FCNN_short_batch(torch.nn.Module):
    def __init__ (self, n_classes):
        super(FCNN_short_batch,self).__init__()
        self.n_classes = n_classes

        self.C1 = nn.Conv2d(1,100,7,padding=1)
        self.C2 = nn.Conv2d(100,150,5,padding=1)
        self.C3 = nn.Conv2d(150,200,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.maxpool2 = nn.MaxPool2d(3,1)
        
        self.BN100 = nn.BatchNorm2d(100)
        self.BN150 = nn.BatchNorm2d(150)
        self.BN200 = nn.BatchNorm2d(200)

        self.fc_cv1 =  nn.Conv2d(200,128,1,padding=0)
        self.fc_cv2 = nn.Conv2d(128,self.n_classes,1,padding=0)
        self.dropout = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        
        x = self.C1(x)
        x = self.BN100(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C2(x)
        x = self.BN150(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C3(x)
        x = self.BN200(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = F.relu(self.dropout(self.fc_cv1(x)))
        x = self.fc_cv2(x)
        x = self.avgpool(x)

        x = x.view([-1,self.n_classes])

        return x
    
    
#Defining an FCNN
class FCNN_short_batch_(torch.nn.Module):
    def __init__ (self, n_classes):
        super(FCNN_short_batch,self).__init__()
        self.n_classes = n_classes

        self.C1 = nn.Conv2d(1,100,7,padding=1)
        self.C2 = nn.Conv2d(100,150,5,padding=1)
        self.C3 = nn.Conv2d(150,200,3,padding=1)
        self.C4 = nn.Conv2d(200,250,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.maxpool2 = nn.MaxPool2d(3,1)
        
        self.BN100 = nn.BatchNorm2d(100)
        self.BN150 = nn.BatchNorm2d(150)
        self.BN200 = nn.BatchNorm2d(200)
        self.BN250 = nn.BatchNorm2d(250)

        self.fc_cv1 =  nn.Conv2d(250,128,1,padding=0)
        self.fc_cv2 = nn.Conv2d(128,self.n_classes,1,padding=0)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        
        x = self.C1(x)
        x = self.BN100(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C2(x)
        x = self.BN150(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.C3(x)
        x = self.BN200(x)
        x = F.relu(x)
        x =self.maxpool1(x)
        x = self.C4(x)
        x = self.BN250(x)
        x = F.relu(x)
        x =self.maxpool2(x)

        x = F.relu(self.dropout(self.fc_cv1(x)))
        x = self.fc_cv2(x)
        x = self.avgpool(x)

        x = x.view([-1,self.n_classes])

        return x
    
class Autoencoder_conv(nn.Module):
    def __init__(self, n_channels):
        super(Autoencoder_conv, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(n_channels, 256, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=(2,3), stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2)
        


    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)
        return x
    
class Autoencoder_conv_(nn.Module):
    def __init__(self, n_channels):
        super(Autoencoder_conv, self).__init__()
        # encoder
        self.enc1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, n_channels, kernel_size=(15,11), padding=1)


    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.out(x))
        return x
       
#Defining an CNN
class CNN_short_fc_comp(torch.nn.Module):
    def __init__ (self, n_classes):
        super(CNN_short_fc_comp,self).__init__()
        self.n_classes = n_classes

        self.C1 = nn.Conv2d(4,100,7,padding=1)
        self.C2 = nn.Conv2d(100,150,5,padding=1)
        self.C3 = nn.Conv2d(150,200,3,padding=1)
        
        self.maxpool1 = nn.MaxPool2d(3,2)
        self.maxpool2 = nn.MaxPool2d((12,4),1)


        self.fc_cv1 =  nn.Linear(200,128)
        self.fc_cv2 = nn.Linear(128,self.n_classes)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x1, x2):
        
        if x2 is not None:
            x1 = self.C1(x1)
            x1 = F.relu(x1)
            x1 = self.maxpool1(x1)
            x1 = self.C2(x1)
            x1 = F.relu(x1)
            x1 = self.maxpool1(x1)
            x1 = self.C3(x1)
            x1 = F.relu(x1)
            x1 = self.maxpool2(x1)
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            x1 = F.relu(self.dropout(self.fc_cv1(x1)))
            x1 = self.fc_cv2(x1)
            
            x2 = self.C1(x2)
            x2 = F.relu(x2)
            x2 = self.maxpool1(x2)
            x2 = self.C2(x2)
            x2 = F.relu(x2)
            x2 = self.maxpool1(x2)
            x2 = self.C3(x2)
            x2 = F.relu(x2)
            x2 =self.maxpool2(x2)
            x2 = self.avgpool(x2)
            x2 = torch.flatten(x2, 1)
            x2 = F.relu(self.dropout(self.fc_cv1(x2)))
            x2 = self.fc_cv2(x2)

            x = torch. sub(x1, x2)
            x = x.view([-1,self.n_classes])

            return x 
        
        else:
            
            x1 = self.C1(x1)
            x1 = F.relu(x1)
            x1 = self.maxpool1(x1)
            x1 = self.C2(x1)
            x1 = F.relu(x1)
            x1 = self.maxpool1(x1)
            x1 = self.C3(x1)
            x1 = F.relu(x1)
            x1 =self.maxpool2(x1)
            
            x1 = self.avgpool(x1)
            x = torch.flatten(x1, 1)
            x = F.relu(self.dropout(self.fc_cv1(x)))
            x = self.fc_cv2(x)
            return x  

        
class FC_Resnet_comp_(ResNet):
    def __init__(self, num_layers = 1, num_classes=1000, pretrained=False, model_size = '18', **kwargs):

        # Start with standard resnet defined here
        if model_size == '18':
            super().__init__(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes, **kwargs)
        elif model_size == '34':
            super().__init__(block = BasicBlock, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50':
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101':
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        elif model_size == '152':
            super().__init__(block = Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        elif model_size == '50_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = num_classes, **kwargs)
        elif model_size == '101_2':
            kwargs['width_per_group'] = 64 * 2
            super().__init__(block = Bottleneck, layers = [3, 4, 23, 3], num_classes = num_classes, **kwargs)
        else:
            assert print("Error: not accepted size. Must be one of '18','34','50','101','152','50_2','101_2' ")
        #if pretrained:
        #    state_dict = torch.hub.load_state_dict_from_url( models.resnet.model_urls["resnet18"], progress=True)
        #    self.load_state_dict(state_dict)

        self.num_classes = num_classes
        
        #    adjust the first conv layer to accpet monochrome input
        self.conv1 = nn.Conv2d(num_layers, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Convert the original fc layer to a convolutional layer.
        #self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        #self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        #self.last_conv.bias.data.copy_ (self.fc.bias.data)

        self.dropout = nn.Dropout(0.2)

    # Reimplementing forward pass.
    def _forward_impl(self, x1, x2):
        
        if x2 is not None:
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)

            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            x1 = self.dropout(x1)
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            x1 = self.fc(x1)
            
            x2 = self.conv1(x2)
            x2 = self.bn1(x2)
            x2 = self.relu(x2)
            x2 = self.maxpool(x2)

            x2 = self.layer1(x2)
            x2 = self.layer2(x2)
            x2 = self.layer3(x2)
            x2 = self.layer4(x2)
            x2 = self.dropout(x2)
            x2 = self.avgpool(x2)
            x2 = torch.flatten(x2, 1)
            x2 = self.fc(x2)

            x = torch. sub(x1, x2)
            x = x.view([-1,self.num_classes])

            return x 
        
        else:
            
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)

            x1 = self.layer1(x1)
            x1 = self.layer2(x1)
            x1 = self.layer3(x1)
            x1 = self.layer4(x1)
            x1 = self.dropout(x1)
            x1 = self.avgpool(x1)
            x1 = torch.flatten(x1, 1)
            x = self.fc(x1)
            return x  
        
        
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self._forward_impl(x1,x2)


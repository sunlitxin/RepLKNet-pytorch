import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
gamma_g = 0.000

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False


        global bes
        bes = 512 * block.expansion * self.fc_scale#25088
        global bes1
        bes1 = 512 * block.expansion #512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)
    def forward(self, x):#   self-attention model----------------------------------------
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)#B X C X W X H
            x = self.bn1(x)#B X C X W X H 128 64 112 112
            x = self.prelu(x)
            x = self.layer1(x)#B X C X W X H 128 64 56 56
            x = self.layer2(x)#B X C X W X H 128 128 28 28
            x = self.layer3(x)#B X C X W X H 128 256 14 14
            x = self.layer4(x)#B X C X W X H 128 512 7 7
        x = x.float() if self.fp16 else x
        return x    ###################  128X512X7X7

#   self-attention model
gamma_g :float

class Self_Atten(torch.nn.Module):
    def __init__(self,fp16=True):
        super(Self_Atten,self).__init__()
        self.fp16 = fp16
        self.conv1_ = nn.Conv2d(3, 512, kernel_size=16, stride=16, bias=False)
        # -----self attention----
        in_dim = 512
        out_dim = 512
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//8, kernel_size=1, )
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//8, kernel_size=1, )
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, )
        self.softmax = nn.Softmax(dim=-1)  #

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,inputs_x):####3.8483
        with torch.cuda.amp.autocast(self.fp16):
            # self-attention-----
            self.x = self.conv1_(inputs_x)#7.4036    7.3486

            m_batchsize, C, width, height = self.x.size()

            proj_query = self.query_conv(self.x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N) 0.4368   0.4382
            proj_key = self.key_conv(self.x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)-0.1255 -0.1249
            if self.fp16:
                proj_query.float()
                proj_key.float()
                energy = torch.bmm(proj_query, proj_key)  # transpose check 1.6250 1.6318
            energy = energy.clamp(-1, 1)
            attention = self.softmax(energy)  # BX (N) X (N)
            proj_value = self.value_conv(self.x).view(m_batchsize, -1, width * height)  # B X C X N  0.1028  1.0370
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # -0.4512
            out = out.view(m_batchsize, C, width, height)
        return out

class ConcatNet_(nn.Module):
    def __init__(self,net1,net2,issamodel=True):
        super(ConcatNet_, self).__init__()
        self.bes = bes
        self.bes1 = bes1
        self.num_features = 512
        self.bn2 = nn.BatchNorm2d(self.bes1, eps=1e-05)
        self.fc = nn.Linear(self.bes, self.num_features)#bes = 512 * block.expansion * self.fc_scale
        self.features = nn.BatchNorm1d(self.num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.net1 = net1
        self.net2 = net2
        self.issamodel = issamodel
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        if self.issamodel:
            out1 = self.net1(x)
            out2 = self.net2(x)
            out = out1 + self.gamma * out2
        else:
            out = self.net1(x)

        x = self.bn2(out)# --- -0.1815
        x = torch.flatten(x, 1)  # ---- -0.1815
        x = self.dropout(x)#
        x = self.fc(x.float() )#-0.8090
        x = self.features(x)#-1.3932
        global gamma_g
        gamma_g = self.gamma
        return x

def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

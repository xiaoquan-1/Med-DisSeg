import math
import torch
import torch.nn as nn
from network.resnet import resnet50
import torch.nn.functional as F
from network.dispersive_loss_implementation import DispersiveLoss, DispersiveLossIntegration, get_dispersive_config
class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
class ELATttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ELATttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 降维
            TeLU(),  # 使用 TeLU 激活函数
            nn.Linear(channels // reduction, channels, bias=False),  # 升维
            nn.Sigmoid()
        )
        # 多尺度空间注意力
        self.spatial_conv1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1, dilation=1)
        self.spatial_conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=2, dilation=2)
        self.spatial_conv3 = nn.Conv2d(channels, 1, kernel_size=3, padding=3, dilation=3)
        self.spatial_fuse = nn.Conv2d(3, 1, kernel_size=1)
        self.spatial_sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)  # 变换维度
        y = self.fc(y).view(b, c, 1, 1)  # 生成注意力权重
        # 多尺度空间注意力
        s1 = self.spatial_conv1(x)
        s2 = self.spatial_conv2(x)
        s3 = self.spatial_conv3(x)
        s = torch.cat([s1, s2, s3], dim=1)
        s = self.spatial_fuse(s)
        s = self.spatial_sigmoid(s)
        return x * y.expand_as(x) * s.expand_as(x)# 重新权重特征
class CBT(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.telu=TeLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.telu(x)
        return x


class adjust(nn.Module):
    def __init__(self, in_c1,in_c2,in_c3,in_c4, out_c):
        super().__init__()
        self.conv1 = CBT(in_c1, 64, kernel_size=1, padding=0, act=True)
        self.conv2 = CBT(in_c2, 64, kernel_size=1, padding=0, act=True)
        self.conv3 = CBT(in_c3, 64, kernel_size=1, padding=0, act=True)
        self.conv4 = CBT(in_c4, 64, kernel_size=1, padding=0, act=True)
        # 加入ELAT注意力模块
        self.att1 = ELATttention(64)
        self.att2 = ELATttention(64)
        self.att3 = ELATttention(64)
        self.att4 = ELATttention(64)
        self.conv_fuse=nn.Conv2d(4*64, out_c, kernel_size=1, padding=0, bias=False)
        #9分类不需要下面这句
        self.sig = nn.Sigmoid()

    def forward(self, x1,x2,x3,x4):
        '''x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)'''
        #加入ELAT注意力模块
        x1 = self.att1(self.conv1(x1))
        x2 = self.att2(self.conv2(x2))
        x3 = self.att3(self.conv3(x3))
        x4 = self.att4(self.conv4(x4))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        #9分类不需要
        x = self.sig(x)
        return x


class MedDisSeg1(nn.Module):
    def __init__(self, H=256, W=256,use_dispersive_loss=False, dispersive_config=None):
        super().__init__()
        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        """从ResNet50中提取出layer0, layer1, layer2, layer3"""
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]
        # dispersive loss配置
        self.use_dispersive_loss = use_dispersive_loss
        if dispersive_config is None and use_dispersive_loss:
            dispersive_config = get_dispersive_config()
        self.dispersive_loss_integration = (
            DispersiveLossIntegration(**dispersive_config) if use_dispersive_loss else None
        )
        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        self.head=adjust(64,256,512,1024,1)
        #九分类self.head=adjust(64,256,512,1024,self.num_classes)




    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]
        # dispersive loss计算（在backbone输出后，x4）
        # 确保始终返回 5 个值（dispersive_loss 可能是 None）
        dispersive_loss = None
        if self.use_dispersive_loss and hasattr(self, 'dispersive_loss_integration'):
            dispersive_loss = DispersiveLoss.compute_dispersive_loss(
                x4.view(x4.size(0), -1),
                loss_type=self.dispersive_loss_integration.dispersive_loss_type,
                temperature=self.dispersive_loss_integration.dispersive_loss_temperature
        )
        x1 = self.up_2x2(x1)
        x2 = self.up_4x4(x2)
        x3 = self.up_8x8(x3)
        x4 = self.up_16x16(x4)

        #pred
        pred=self.head(x1,x2,x3,x4)

        return pred, dispersive_loss  # 返回 dispersive_loss
    def extract_features(self, image):
        """只提取 x4 特征，用于可视化"""
        x0 = image
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return x4.detach()



if __name__ == "__main__":
    model = MedDisSeg1().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)

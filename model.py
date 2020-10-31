import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from resnet import resnet50


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ----model---- #

class PCBModel(nn.Module):
    def __init__(self, last_conv_stride=1, last_conv_dilation=1, num_stripes=6, local_conv_out_channels=256, num_classes=1502):
        super(PCBModel, self).__init__()

        self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride, last_conv_dilation=last_conv_dilation)
        self.num_stripes = num_stripes

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(nn.Conv2d(2048, local_conv_out_channels, 1), nn.BatchNorm2d(local_conv_out_channels), nn.ReLU(inplace=True)))

        if num_classes > 0:
            self.fc_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)

    def forward(self, x):
        """
        Returns:
          local_feat_list: each member with shape [N, c]
          logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        # in case stripe_h is able to be devide by several parts
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                (stripe_h, feat.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        if hasattr(self, 'fc_list'):
            return local_feat_list, logits_list

        return local_feat_list

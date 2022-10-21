# +
# reference: https://towardsdatascience.com/biomedical-image-segmentation-unet-991d075a3a4b
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch.nn as nn
import torch

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Nested_3DUNet(nn.Module):

    def __init__(self, in_ch=3, n_class=4 ):
        super(Nested_3DUNet, self).__init__()
        
        out_ch = n_class

        n1 = 32#64
        filters = [n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32] # Recently doubled filters for more parameters

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # backbone
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

        self.dropoutx0_0 = nn.Dropout3d(p=0.1)
        self.dropoutx1_0 = nn.Dropout3d(p=0.2)
        self.dropoutx2_0 = nn.Dropout3d(p=0.3)
        self.dropoutx3_0 = nn.Dropout3d(p=0.4)
        self.dropoutx4_0 = nn.Dropout3d(p=0.5)

        self.dropoutx0_4 = nn.Dropout3d(p=0.1)
        self.dropoutx1_3 = nn.Dropout3d(p=0.2)
        self.dropoutx2_2 = nn.Dropout3d(p=0.3)
        self.dropoutx3_1 = nn.Dropout3d(p=0.4)        

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x0_0_drop = self.dropoutx0_0(self.pool(x0_0))
        x1_0 = self.conv1_0(x0_0_drop)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x1_0_drop = self.dropoutx1_0(self.pool(x1_0))
        x2_0 = self.conv2_0(x1_0_drop)        
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x2_0_drop = self.dropoutx2_0(self.pool(x2_0))
        x3_0 = self.conv3_0(x2_0_drop)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x3_0_drop = self.dropoutx3_0(self.pool(x3_0))
        x4_0 = self.conv4_0(x3_0_drop)
        x4_0_drop = self.dropoutx0_4(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0_drop)], 1))
        x3_1 = self.dropoutx3_1(x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x2_2 = self.dropoutx2_2(x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x1_3 = self.dropoutx1_3(x1_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        x0_4 = self.dropoutx0_4(x0_4)

        output = self.final(x0_4)
        return output

"""
# model is defined below in Select Model

from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = Nested_3DUNet(in_ch=1, n_class=2)
model = model.to(device)
summary(model, input_size = (1,16,64,64))
"""

# +
# from torchsummary import summary
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
# model = Nested_3DUNet(in_ch=1, n_class=4)
# model = model.to(device)
# summary(model, input_size = (1,16,64,64))# NCDHW
# -



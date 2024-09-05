import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation,PointResnet,PointNetSetAbstractionMsgAttention


####偏制注意力
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsgAttention(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa4 = PointNetSetAbstractionMsg(256, [0.2, 0.4,0.6], [64, 96,128], 320,[ [64, 196], [ 96, 256],[128,256]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 708, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3,mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 1024, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=708 + 256, mlp=[256, 128])
        self.fp4 = PointNetFeaturePropagation(in_channel=320 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150 + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.conv_fuse = nn.Sequential(nn.Conv1d(640, 128, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.LeakyReLU(negative_slope=0.2))
        ################
        self.SL = SA_Layer(128)
        self.SL1 = SA_Layer(320)
        self.SL2 = SA_Layer(512)
        self.SL3 = SA_Layer(1024)

        self.PR1 = PointResnet(0.4,64,320,320,640,320)
        self.PR2 = PointResnet(0.5, 64, 708, 708, 904, 708)
        self.PR3 = PointResnet(0.6,64,512,512,1024,512)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_xyz, l1_points = self.PR1(l1_xyz, l1_points)
        #######
        # S1_points = self.SL1(l1_points)
        # S2_points = self.SL1(S1_points)
        # S3_points = self.SL1(S2_points)
        # l1_points = torch.cat((S3_points,l1_points),dim=1)
        l4_xyz, l4_points = self.sa4(l1_xyz, l1_points)
        l4_xyz, l4_points = self.PR2(l4_xyz, l4_points)


        l2_xyz, l2_points = self.sa2(l4_xyz, l4_points)
        l2_xyz, l2_points = self.PR3(l2_xyz, l2_points)
        ########
        # S1_points = self.SL2(l2_points)
        # S2_points = self.SL2(S1_points)
        # S3_points = self.SL2(S2_points)
        # l2_points = torch.cat((S3_points, l2_points), dim=1)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #######
        #### l3_points = self.SL3(l3_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l4_points = self.fp2(l4_xyz, l2_xyz, l4_points, l2_points)
        l1_points = self.fp4(l1_xyz, l4_xyz, l1_points, l4_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)

        S1_points = self.SL(l0_points)
        S2_points = self.SL(S1_points)
        S3_points = self.SL(S2_points)
        S4_points = self.SL(S3_points)
        S_points = torch.cat((S1_points, S2_points, S3_points, S4_points, l0_points), dim=1)
        l0_points = self.conv_fuse(S_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction,PointResnet,PointNetSetAbstractionMsgAttention
import torch

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
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 128,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa4 = PointNetSetAbstractionMsg(256, [0.2, 0.4,0.6], [64, 96,128], 320, [[64, 196], [96, 256],[128,256]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 708,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        self.PR1 = PointResnet(0.4, 64, 320, 320, 640, 320)
        self.PR2 = PointResnet(0.5, 64, 708, 708, 1416, 708)
        self.PR3 = PointResnet(0.6, 64, 640, 640, 1280, 640)
        
        self.SL1 = SA_Layer(128)
        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.conv_fuse0 = nn.Sequential(nn.Conv1d(6, 128, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_points = torch.cat((xyz,norm), dim=1)
        l1_points = self.conv_fuse0(l1_points)
        S1_points = self.SL1(l1_points)
        S2_points = self.SL1(S1_points)
        S3_points = self.SL1(S2_points)
        S_points = torch.cat((S1_points, S2_points, S3_points, l1_points), dim=1)
        l1_points = self.conv_fuse(S_points)
        l1_xyz, l1_points = self.sa1(xyz, l1_points)
        l1_xyz, l1_points = self.PR1(l1_xyz, l1_points)

        l4_xyz, l4_points = self.sa4(l1_xyz, l1_points)
        l4_xyz, l4_points = self.PR2(l4_xyz, l4_points)

        l2_xyz, l2_points = self.sa2(l4_xyz, l4_points)
        l2_xyz, l2_points = self.PR3(l2_xyz, l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss



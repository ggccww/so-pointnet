import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
"""
打印时间
"""
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()
"""
pc_normalize为point cloud normalize，即将点云数据进行归一化处理
"""
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
"""
square_distance函数
该函数主要用来在ball query过程中确定每一个点距离采样点的距离。

函数输入是两组点，N为第一组点src个数，M为第二组点dst个数，C为输入点的通道数，返回的是两组点之间两两的欧几里德距离，即N × M 矩阵。

由于在训练中数据通常是以Mini-Batch的形式输入的，所以有一个Batch数量的维度为B。
"""
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

"""
5.按照输入的点云数据和索引返回索引的点云数据
例如points为B*2084*3点云，idx为【5，666，1000，2000】则返回batch中的第5，666，1000,2000个点组成的B*4*3的点云集

如果idx为一个【B,D1,...DN]则它会按idx的维度结构将其提取成[B,D1,...DN,C]
"""
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

"""
6.farthest_point_sample函数是来自于Pointnet++的FPS(Farthest Point Sampling) 最远点采样法
该方法比随机采样的优势在于它可以尽可能的覆盖空间中的所有点。
最远点采样是Set Abstraction模块中较为核心的步骤，其目的是从一个输入点云中按照所需要的点的个数npoint采样出足够多的点，并且点与点之间的距离要足够远。最后的返回结果是npoint个采样点在原始点云中的索引。

假设一共有n个点,整个点集为N = {f1, f2,…,fn}, 目标是选取npoint个起始点做为下一步的中心点:
（1）随机选取一个点fi为起始点，并写入起始点集 B = {fi};
（2）选取剩余n-1个点计算和fi点的距离(这就用到4定义的函数），选择最远点fj写入起始点集B={fi,fj};
（3）选取剩余n-2个点计算和点集B中每个点的距离, 将最短的那个距离作为该点到点集的距离,
（4）这样得到n-2个到点集的距离，选取最远的那个点写入起始点B = {fi, fj ,fk},同时剩下n-3个点, 如果n1=3， 则到此选择完毕;
（5）如果n1 > 3则重复上面步骤直到选取npoint个起始点为止.
"""
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

"""
7.query_ball_point函数用于寻找球形领域中的点
输入中radius为球形领域的半径，nsample为每个领域中要采样的点，
new_xyz为S个球形领域的中心（由最远点采样在前面得出），xyz为所有的点云；
输出为每个样本的每个球形领域的nsample个采样点集的索引[B,S,nsample]。
"""
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

"""8.Sampling+Grouping主要用于将整个点云分散成局部的group
对于每一个group都可以用PointNet单独的提取局部的全局特征
Sampling+Grouping分成了sampl_and_group和sampl_and_group_all两个函数
其区别在于sample_and_group_all直接将所有点作为一个group

例如：
512=npoint:poins sampled in farthest point sampling

0.2=radius:search radius in local region

32=nsample:how many points in each local region
将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征

（注意xyz和poinets的区别）"""
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

"""9.sample_and_group_all
直接将所有点作为一个group，增加一个长度为1的维度，npoint = 1
如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征"""
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

"""10.Sampling + Grouping + PointNet组合层 (SetAbstraction层)
PointNetSetAbstraction类实现普通的SetAbstraciton：
然后通过sample_and_group的操作形成局部group
然后对局部group中的每一个点做MLP操作，最后进行局部最大池化，得到局部的全局特征"""
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointResnet(nn.Module):
    def __init__(self,  radius, nsample, in_channel, mlp, P1,P2):
        super(PointResnet, self).__init__()
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channel+3
        out_channel = mlp
        self.conv = nn.Conv2d(last_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)


        self.conv1 = nn.Conv1d(P2,P1 , 1)
        self.bn1 = nn.BatchNorm1d(P1)
        self.conv2 = nn.Conv1d(P1, P2, 1)
        self.bn2 = nn.BatchNorm1d(P2)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            Respoints = points
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        K = self.nsample
        radius = self.radius
        group_idx = query_ball_point(radius, K, xyz, xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz -= xyz.view(B, N, 1, C)
        grouped_points = index_points(points, group_idx)
        grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        grouped_points = F.relu(self.bn(self.conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]
        new_xyz = xyz.permute(0,2,1)
        new_points = F.relu(self.bn1(self.conv1(new_points)))
        new_points = self.conv2(new_points)
        new_points = self.bn2(new_points)
        new_points += Respoints
        new_points = F.relu(new_points)


        return new_xyz, new_points

"""11.PointNetSetAbstractionMsg类实现MSG方法的Set Abstraction
这里radius_list输入的是一个list，例如[0.1,0.2,0.4]
对于不同的半径做ball query，将不同半径下的点云特征保存在new_points_list中，最后再拼接到一起."""
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

"""Feature Propagation的实现主要通过线性差值和MLP完成（用在分割，要做上采样）
PointNet++会随着网络逐层降低采样的点数，这样来保证网络获得足够的全局信息，但是这样会导致无法完成分割任务，因为分割任务是一个端到端的，必须保证输出与输入点数相同。
一种完成分割任务的方法就是不下采样点，始终将所有点放入网络进行计算。但这样需要消耗大量计算成本。另一种比较常用的方法就是进行插值了，利用已知点，来补足需要点。
FP模块的目的就是利用已知的特征点来进行插值，使网络输出与输入点数相同的特征。具体的做法可见下一步。

当点的个数只有一个的时候，采用repeat直接复制成N个点
当点的个数大于一个的时候，采用线性差值的方式进行上采样
拼接上下采样对应点的SA的特征，再对拼接后的每一个点做一个MLP
实现主要通过线性差值与MLP堆叠完成，距离越远的点权重越小，最后对于每一个点的权重再做一个全局的归一化。"""
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        """
                Input:
                    利用前一层的点对后面的点进行插值
                    xyz1: input points position data, [B, C, N]  l2层输出 xyz
                    xyz2: sampled input points position data, [B, C, S]  l3层输出  xyz
                    points1: input points data, [B, D, N]  l2层输出  points
                    points2: input points data, [B, D, S]  l3层输出  points

                Return:
                    new_points: upsampled points data, [B, D', N]
                """
        "  将B C N 转换为B N C 然后利用插值将高维点云数目S 插值到低维点云数目N (N大于S)"
        "  xyz1 低维点云  数量为N   xyz2 高维点云  数量为S"


        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            "如果最后只有一个点，就将S直复制N份后与与低维信息进行拼接"
            interpolated_points = points2.repeat(1, N, 1)
        else:
            "如果不是一个点 则插值放大 128个点---->512个点"
            "此时计算出的距离是一个矩阵 512x128 也就是512个低维点与128个高维点 两两之间的距离"
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            "找到距离最近的三个邻居，这里的idx：2,512,3的含义就是512个点与128个距离最近的前三个点的索引，" \
            "例如第一行就是：对应128个点中那三个与512中第一个点距离最近"
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            "对dist_recip的倒数求和 torch.sum   keepdim=True 保留求和后的维度  2,512,1"
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            """
                        这里的weight是计算权重  dist_recip中存放的是三个邻居的距离  norm中存放是距离的和  
                        两者相除就是每个距离占总和的比重 也就是weight
                        """
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            """
            points2: 2,128,256 (128个点 256个特征)   idx 2,512,3 （512个点中与128个点距离最近的三个点的索引）
            index_points(points2, idx) 从高维特征（128个点）中找到对应低维特征（512个点） 对应距离最小的三个点的特征 2,512,3,256
            这个索引的含义比较重要，可以再看一下idx参数的解释，其实2,512,3,256中的512都是高维特征128个点组成的。
            例如 512中的第一个点 可能是由128中的第 1 2 3 组成的；第二个点可能是由2 3 4 三个点组成的
            -------------------------------------------
            weight: 2,512,3    weight.view(B, N, 3, 1) ---> 2,512,3,1
            a与b做*乘法，原则是如果a与b的size不同，则以某种方式将a或b进行复制，使得复制后的a和b的size相同，然后再将a和b做element-wise的乘法。
            这样做乘法就相当于 512,3,256  中的三个点的256维向量都会去乘它们的距离权重，也就是一个数去乘256维向量
            torch.sum dim=2 最后在第二个维度求和 取三个点的特征乘以权重后的和 也就完成了对特征点的上采样
            """

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


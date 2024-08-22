# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "model"))
import common
import torch
import torch.nn as nn
import pdb


def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        """
        global average pooling: feature --> point
        
        nn.AdaptiveAvgPool2d(1)
        自适应平均二维池化，不用指定池化核的大小，指定输出的大小，自动反推池化核大小
        1 是 output size。要求输入二元元组（h，w），仅输入一个数字1，效果就是长宽都是1，（1， 1）
        
        池化操作在通道维度上进行
        在这里，每个通道都有一个计算结果，依据这些结果，计算出每个通道的权重，权重之和为1
        """
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        """
        在最后一个维度上，进行softmax计算
        """
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        """
        proj : projection 投影，映射
        """
        proj_query = x.view(m_batchsize, N, -1)
        """
        permute : 置换、排列、交换
        
        转置，为后面的bmm操作做准备
        """
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        """
        计算机query和key之间的相关矩阵
        bmm：批量矩阵相乘
        
        最后结果：B*N*N
        """
        energy = torch.bmm(proj_query, proj_key)
        """
        模块的softmax在最后一个维度进行计算
        torch.max第二个参数指定在哪个维度上进行计算
        -1，代表最后一个维度，与softmax对应
        
        torch.max：返回指定维度上的最大值和对应索引
        torch.max(~)[0]：取得指定维度最大值
        keepdim=True：保持维度数不变，也即是结果形状为B*N*1
        
        expand_as(energy)：将结果扩展至与energy形状相同，也就是最后一个维度自我复制，B*N*N
        """
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        """
        转换权重矩阵
        """
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim
        """
        1、三维卷积
        通过三维卷积获得通道和空间特征Wcsa
        
        2、sigmoid
        计算Wcsa：σ（Wcsa）
        计算细节：输出张量与输入张量形状一致，每个元素被压缩到0~1之间
        
        3、gamma
        比例因子gamma
        gamma*σ（Wcsa）
        
        4、整体计算公式
        Fn输入csam模块的特征
        gamma*σ（Wcsa）*Fn + Fn
        """
        
        """
        3, 1, 1：输出与输入形状一致
        
        ？？？输入输出的通道数都是1，但是特征图通道不是1
        ？？？与lam类似？：将通道维度往后移一位，以第二个维度，存放各rg和最后卷积层的共十一个输出，经过特定卷积层整合一个结果，第二维为1
        """
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        """        
        nn.Parameter将输入的张量转化为模型的参数，在学习中会被更新
        """
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        """
        out = x.unsqueeze(1)
        与lam相似的操作：通道维度延后，第二个维度另作他用：计算Wcsa
        第二个维度为1，刚好符合三维卷积输入1的要求
        """
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        """
        实际效果：去除之前x.unsqueeze(1)，添加的第二个维度，回复原形状
        """
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        """
        ???
        在csam的处理中，输出与输入的形状一致，添加的第二个维度，最终去除了；
        但是与lam的输出（添加的第二个维度没有删除）不一致，后续cat操作
        """
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        """
        python内置函数super，返回父类的临时对象
        super的两个参数：子类名称，子类实例
        调用父类的初始化函数__init__，继承父类的属性和方法（为什么使用方法，不太理解，方法不是在__init__函数中定义）
        """

        modules_body = []
        """
        循环最终效果：卷积+relu+卷积
        """
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            # if bn: print("bn")
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, bn):
        super(ResidualGroup, self).__init__()
        """
        限制
        固定值：act=nn.ReLU(True), res_scale=1
        输入的这两个值无效
        """
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=bn, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Holistic Attention Network (HAN)
class HAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAN, self).__init__()

        # bn = args.bn

        # n_resgroups = args.n_resgroups
        # n_resblocks = args.n_resblocks
        # n_feats = args.n_feats
        # kernel_size = 3
        # """
        # rcan : reduction控制特征图的减少
        # """
        # reduction = args.reduction
        # """
        # scale是定量
        # 模型最后的上采样模型是确定的
        # ？那怎么做到同一模型，实现不同scale的提升
        # """
        # scale = args.scale[0]
        # act = nn.ReLU(True)


        bn = False

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 128
        kernel_size = 3
        reduction = 16
        act = nn.ReLU(True)

        res_scale = 1

        n_colors = 3


        """
        修改之后，HAN模型默认不添加shift_mean模块
        
        现在添加
        """
        # self.shift_mean = False
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.shift_mean = True
        rgb_mean=(0.485, 0.456, 0.406)
        rgb_std=(0.229, 0.224, 0.225)
        rgb_range = 255
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)


        # define head module
        # modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, bn=bn) \
        #     for _ in range(n_resgroups)]
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks, bn=bn) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]
        modules_tail = [conv(n_feats, n_colors, kernel_size)]



        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        """
        n_feats*11 : n_rgs=10 + rgs之后的一个卷积层
        与论文所展示结构略有区别：将最后卷积层的输出，也输入到lam中
        """
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        """
        整合lam和csam的输出
        """
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if self.shift_mean:
            x = self.sub_mean(x)

        x = self.head(x)
        res = x
        #pdb.set_trace()
        """
        属性_modules以OrderDict的形式，返回所有子模块
        字典方法items()，返回包含字典中所有（key, value）元组的列表
        """
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            """
            unsqueeze在指定位置，增加一个大小为1的维度，创建源tensor的视图，输入张量与输出张量共享内存，改变输出张量，输入张量也会变化
            unsqueeze(1) 中的”1“指的是第二个位置
            batch, channel, height, width, depth
            新增加的维度1，代表num_midlayer
            占据了原先的channel通道，在卷积中，当作channel处理
            """
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                """
                在维度1上，拼接两个张量
                """
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        """
        out1:rgs的输出
        out2:lam的输出
        经过self.last_conv的处理：输入11*feat通道，输出1*feat通道，整合各层次之间的信息
        """
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        """
        out1：经过rgs和一个卷积层得到输出，将输出输入到csam中，得到out1
        """
        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)

        # test
        # print(f"out1 : {out1.shape}")
        # print(f"out2 : {out2.shape}")

        """      
        通过卷积操作，缩小一倍第二维度，从2*feat到1*feat
        这样结果就与最初的卷积层输出形状一致，可以完成元素相加        
        """
        res = self.last(out)
        """
        res：long skip、lam和csam的整合结果
        """
        res += x

        x = self.tail(res)

        if self.shift_mean:
            x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
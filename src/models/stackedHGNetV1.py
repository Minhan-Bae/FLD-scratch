import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

import torch
import torch.nn as nn


class AddCoordsTh(nn.Module):
    def __init__(self, x_dim, y_dim, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).to(input_tensor)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).to(input_tensor)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :],
                                        0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel).to(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel>0.05,
                                              xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel>0.05,
                                              yy_channel, zero_tensor)
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)


        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            ret = torch.cat([ret, xx_boundary_channel,
                             yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, with_boundary,
                 in_channels, out_channels, first_one=False, relu=False, bn=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r,
                                    with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        self.with_boundary = with_boundary
        self.first_one = first_one


    def forward(self, input_tensor, heatmap=None):
        assert (self.with_boundary and not self.first_one) == (heatmap is not None)
        ret = self.addcoords(input_tensor, heatmap)
        ret = self.conv(ret)
        if self.bn is not None:
            ret = self.bn(ret)
        if self.relu is not None:
            ret = self.relu(ret)

        return ret


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1).to(input_tensor)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).to(input_tensor)

        xx_channel = xx_channel / (x_dim - 1)
        yy_channel = yy_channel / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

def _make_grid(h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h).float() / (h-1)*2-1,
        torch.arange(w).float() / (w-1)*2-1)
    return yy, xx


def get_coords_from_heatmap(heatmap):
    """
    inputs:
    - heatmap: batch x npoints x h x w
    outputs:
    - coords: batch x npoints x 2 (x,y), [-1, +1]
    - radius_sq: batch x npoints
    """
    batch, npoints, h, w = heatmap.shape

    yy, xx = _make_grid(h, w)
    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum # batch x npoints
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum # batch x npoints
    coords = torch.stack([xx_coord, yy_coord], dim=-1)

    return coords


class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'


class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size-1)//2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MultiViewBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, groups=1):
        super(MultiViewBlock, self).__init__()

        assert out_dim % 4 == 0
        dim1 = out_dim // 2
        dim2 = out_dim // 4

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, dim1, 3, relu=False)
        self.bn2 = nn.BatchNorm2d(dim1)
        self.conv2 = ConvBlock(dim1, dim2, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(dim2)
        self.conv3 = ConvBlock(dim2, dim2, 3, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x

        # inp_dim x mid_dim
        out1 = self.bn1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        # mid_dim x mid_dim
        out2 = self.bn2(out1)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        # mid_dim x out_dim
        out3 = self.bn3(out2)
        out3 = self.relu(out3)
        out3 = self.conv3(out3)

        out = torch.cat([out1, out2, out3], dim=1)

        out += residual
        return out


class ResBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim=None):
        super(ResBlock, self).__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_dim)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0, up_mode='nearest', 
                 add_coord=False, pool_type="origin", use_multiview=False, 
                 first_one=False, x_dim=64, y_dim=64):
        super(Hourglass, self).__init__()
        nf = f + increase

        if use_multiview:
            Block = MultiViewBlock
        else:
            Block = ResBlock

        if add_coord:
            self.coordconv = CoordConvTh(x_dim=x_dim, y_dim=y_dim,
                                         with_r=True, with_boundary=True,
                                         relu=False, bn=False,
                                         in_channels=f, out_channels=f,
                                         first_one=first_one,
                                         kernel_size=1,
                                         stride=1, padding=0)
        else:
            self.coordconv = None
        self.up1 = Block(f, f)

        # Lower branch
        if pool_type == "origin":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "blur":
            self.pool1 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
                                         BlurPool(f, filt_size=3, stride=2)])
            #self.pool1 = BlurPool(f, filt_size=3, stride=2)
        else:
            assert False

        self.low1 = Block(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n=n-1, f=nf, increase=increase, up_mode=up_mode, add_coord=False, pool_type=pool_type)
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        super().__init__()

        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        # pn x en x 1 x 1.
        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        # some keypoints are not coverred by any edges,
        # in these cases, we must add a constant bias to their heatmap weights.
        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        # pn x 1.
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        # input: batch_size x en x hw x hh.
        # output: batch_size x pn x hw x hh.
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class StackedHGNetV1(nn.Module):
    def __init__(self, classes_num, edge_info,
                 nstack=4, nlevels=4, in_channel=256, increase=0,
                 add_coord=True, pool_type="origin", use_multiview=False):
        super(StackedHGNetV1, self).__init__()

        self.nstack = nstack
        self.add_coord = add_coord
        self.pool_type = pool_type
        self.use_multiview = use_multiview

        self.num_heats = classes_num[0]
        self.num_edges = classes_num[1]
        self.num_points = classes_num[2]
        self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)

        if self.add_coord:
            convBlock = CoordConvTh(x_dim=256, y_dim=256,
                                    with_r=True, with_boundary=False,
                                    relu=True, bn=True,
                                    in_channels=3, out_channels=64,
                                    kernel_size=7,
                                    stride=2, padding=3)
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)

        if pool_type == "origin":
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "blur":
            pool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
                                   BlurPool(128, filt_size=3, stride=2)])
            #pool = BlurPool(128, filt_size=3, stride=2)
        else:
            assert False

        if self.use_multiview:
            Block = MultiViewBlock
        else:
            Block = ResBlock

        self.pre = nn.Sequential(
            convBlock,
            Block(64, 128),
            pool,
            Block(128, 128),
            Block(128, in_channel)
        )

        self.hgs = nn.ModuleList(
            [Hourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, pool_type=self.pool_type, use_multiview=self.use_multiview, first_one=(_==0))
             for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Block(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False)
             for _ in range(nstack)])
        self.out_edgemaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_edges, 1, relu=False, bn=False)
             for _ in range(nstack)])
        self.out_pointmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_points, 1, relu=False, bn=False)
             for _ in range(nstack)])

        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_edgemaps = nn.ModuleList(
            [ConvBlock(self.num_edges, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_pointmaps = nn.ModuleList(
            [ConvBlock(self.num_points, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.nstack = nstack

        self.heatmap_act = Activation("in+relu", self.num_heats)
        self.edgemap_act = Activation("sigmoid", self.num_edges)
        self.pointmap_act = Activation("sigmoid", self.num_points)

        self.inference = False

    def set_inference(self, inference):
        self.inference = inference

    def get_preds_fromhm(self, hm):
        _, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]]).to(preds)
                    preds[i, j] += diff.sign_() * .25

        preds = (preds * 2 + 1) / torch.tensor([64, 64]).to(preds).view(1, 1, 2) - 1
        return preds

    def forward(self, x):
        x = self.pre(x)

        y = []
        heatmaps = None
        for i in range(self.nstack):
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)

            heatmaps0 = self.out_heatmaps[i](feature)
            heatmaps = self.heatmap_act(heatmaps0)
            edgemaps0 = self.out_edgemaps[i](feature)
            edgemaps = self.edgemap_act(edgemaps0)
            pointmaps0 = self.out_pointmaps[i](feature)
            pointmaps = self.pointmap_act(pointmaps0)

            edge_point_attention_mask = self.e2h_transform(edgemaps) * pointmaps
            landmarks = get_coords_from_heatmap(edge_point_attention_mask * heatmaps)

            if i < self.nstack - 1:
                x = x + \
                    self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps) + \
                    self.merge_edgemaps[i](edgemaps) + \
                    self.merge_pointmaps[i](pointmaps)

            y.append(landmarks)
            y.append(edgemaps)
            y.append(pointmaps)

        return y, edge_point_attention_mask, landmarks
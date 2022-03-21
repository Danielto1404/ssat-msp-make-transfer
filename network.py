import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init


####################################################################
# ------------------------- Discriminators --------------------------
####################################################################
class Dis(nn.Cell):
    def __init__(self, input_dim, norm='None'):
        super(Dis, self).__init__()
        ch = 32
        n_layer = 5
        self.model = self._make_net(ch, input_dim, n_layer, norm)

    def _make_net(self, ch, input_dim, n_layer, norm):
        model = []
        model += [ConvNormReLU(input_dim, ch, kernel_size=3, stride=2, padding=1, alpha=0.2, norm_mode=norm)]
        tch = ch
        for i in range(1, n_layer):
            model += [ConvNormReLU(tch, tch * 2, kernel_size=3, stride=2, padding=1, alpha=0.2, norm_mode=norm)]
            tch *= 2
        model += [ConvNormReLU(tch, tch * 2, kernel_size=3, stride=2, padding=1, alpha=0.2, norm_mode=norm)]
        tch *= 2
        model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.SequentialCell(*model)

    def construct(self, x):
        out = self.model(x)
        out = out.reshape(-1)
        return out


class MultiScaleDis(nn.Cell):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None'):
        super(MultiScaleDis, self).__init__()
        ch = 32
        self.downsample = nn.AvgPool2d(3, stride=2)

        self.Diss = nn.CellList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm))

    def _make_net(self, ch, input_dim, n_layer, norm):
        model = []
        model += [ConvNormReLU(input_dim, ch, kernel_size=4, stride=2, padding=1, alpha=0.2, norm_mode=norm)]
        tch = ch
        for _ in range(1, n_layer):
            model += [ConvNormReLU(tch, tch * 2, kernel_size=4, stride=2, padding=1, alpha=0.2, norm_mode=norm)]
            tch *= 2
        model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.SequentialCell(*model)

    def construct(self, x):
        concat = ops.Concat(axis=0)
        iter = 0
        outs = 0
        for Dis in self.Diss:
            out = Dis(x)
            out = out.reshape(-1)
            x = self.downsample(x)
            if iter == 0:
                outs = out
            else:
                outs = concat((outs, out))
            iter += 1
        return outs


####################################################################
# ---------------------------- Encoders -----------------------------
####################################################################

class E_content(nn.Cell):
    def __init__(self, input_dim, ngf=64):
        super(E_content, self).__init__()
        self.layer_1 = ConvNormReLU(in_planes=input_dim, out_planes=ngf, kernel_size=7, stride=1, padding=3,
                                    norm_mode='instance', alpha=0.2)
        self.layer_2 = ConvNormReLU(in_planes=ngf, out_planes=ngf * 2, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)
        self.layer_3 = ConvNormReLU(in_planes=ngf * 2, out_planes=ngf * 4, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)

    def construct(self, x):
        # x (3, 256, 256)
        feature_map1 = self.layer_1(x)
        # x (64, 256, 256)
        feature_map2 = self.layer_2(feature_map1)
        # x (64*2, 128, 128)
        feature_map3 = self.layer_3(feature_map2)
        # x (64*4, 64, 64)
        return feature_map1, feature_map2, feature_map3


class E_makeup(nn.Cell):
    def __init__(self, input_dim, ngf=64):
        super(E_makeup, self).__init__()
        self.layer_1 = ConvNormReLU(in_planes=input_dim, out_planes=ngf, kernel_size=7, stride=1, padding=3,
                                    norm_mode='instance', alpha=0.2)
        self.layer_2 = ConvNormReLU(in_planes=ngf, out_planes=ngf * 2, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)
        self.layer_3 = ConvNormReLU(in_planes=ngf * 2, out_planes=ngf * 4, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)

    def construct(self, x):
        # x (3, 256, 256)
        feature_map1 = self.layer_1(x)
        # x (64, 256, 256)
        feature_map2 = self.layer_2(feature_map1)
        # x (64*2, 128, 128)
        feature_map3 = self.layer_3(feature_map2)
        # x (64*4, 64, 64)
        return feature_map3


class E_semantic(nn.Cell):
    def __init__(self, input_dim, ngf=32):
        super(E_semantic, self).__init__()
        self.layer_1 = ConvNormReLU(in_planes=input_dim, out_planes=ngf, kernel_size=7, stride=1, padding=3,
                                    norm_mode='instance', alpha=0.2)
        self.layer_2 = ConvNormReLU(in_planes=ngf, out_planes=ngf * 2, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)
        self.layer_3 = ConvNormReLU(in_planes=ngf * 2, out_planes=ngf * 4, kernel_size=3, stride=2, padding=1,
                                    norm_mode='instance', alpha=0.2)

    def construct(self, x):
        # x (3, 256, 256)
        feature_map1 = self.layer_1(x)
        # x (32, 256, 256)
        feature_map2 = self.layer_2(feature_map1)
        # x (32*2, 128, 128)
        feature_map3 = self.layer_3(feature_map2)
        # x (32*4, 64, 64)
        return feature_map3


####################################################################
# ----------------------- Feature Fusion (FF) ----------------------
####################################################################

class FeatureFusion(nn.Cell):
    def __init__(self, ngf=64):
        super(FeatureFusion, self).__init__()
        input_dim = (32 * 4 + 64 * 4)
        self.conv1 = ConvNormReLU(in_planes=input_dim, out_planes=ngf * 8, kernel_size=3, stride=2, padding=1,
                                  norm_mode='instance', alpha=0.2)
        self.resize = nn.ResizeBilinear()

    def construct(self, x, y):
        # x[0] (64*1, 256, 256)
        # x[1] (64*2, 128, 128)
        # x[2] (64*4, 64, 64)
        # y (32*4, 64,64)
        B, C, H, W = x[0].shape
        concat = ops.Concat(axis=1)
        out = concat((x[2], y))
        out = self.conv1(out)
        feature_map1 = self.resize(x[0], size=(H // 4, W // 4))
        feature_map2 = self.resize(x[1], size=(H // 4, W // 4))
        feature_map3 = self.resize(x[2], size=(H // 4, W // 4))
        feature_map4 = self.resize(out, size=(H // 4, W // 4))
        feature_map5 = self.resize(y, size=(H // 4, W // 4))
        output = concat((feature_map1, feature_map2, feature_map3, feature_map4, feature_map5))
        return output


####################################################################
# ----------------- SymmetryAttention Moudle -----------------------
####################################################################

class SymmetryAttention(nn.Cell):
    def __init__(self):
        super(SymmetryAttention, self).__init__()
        in_dim = 64 * 17
        self.softmax_alpha = 100
        self.fa_conv = ConvNormReLU(in_dim, in_dim // 8, kernel_size=1, stride=1, padding=0, norm_mode='instance',
                                    alpha=0.2)
        self.fb_conv = ConvNormReLU(in_dim, in_dim // 8, kernel_size=1, stride=1, padding=0, norm_mode='instance',
                                    alpha=0.2)
        self.norm = nn.Norm(axis=1, keep_dims=True)

        self.softmax = nn.Softmax(axis=2)

    def warp(self, fa, fb, a_raw, b_raw, alpha):
        '''
        calculate correspondence matrix and warp the exemplar features
        '''
        n, c, h, w = fa.shape
        _, raw_c, _, _ = a_raw.shape
        bmm = ops.BatchMatMul()  # GPU
        transpose = ops.Transpose()  # GPU
        # subtract mean
        fa = fa - fa.mean(axis=(2, 3), keep_dims=True)
        fb = fb - fb.mean(axis=(2, 3), keep_dims=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.reshape(n, c, -1)
        fb = fb.reshape(n, c, -1)
        fa = fa / self.norm(fa)
        fb = fb / self.norm(fb)

        # correlation matrix, gonna be huge (4096*4096)
        # use matrix multiplication for CUDA speed up
        # Also, calculate the transpose of the atob correlation
        # warp the exemplar features b, taking softmax along the b dimension

        input_perm = (0, 2, 1)
        energy_ab_T = bmm(transpose(fb, input_perm), fa) * alpha
        corr_ab_T = self.softmax(energy_ab_T)  # n*HW*C @ n*C*HW -> n*HW*HW
        b_warp = bmm(b_raw.reshape(n, raw_c, h * w), corr_ab_T)  # n*HW*1
        b_warp = b_warp.reshape(n, raw_c, h, w)

        energy_ba_T = bmm(transpose(fa, input_perm), fb) * alpha
        corr_ba_T = self.softmax(energy_ba_T)  # n*HW*C @ n*C*HW -> n*HW*HW
        #corr_ba_T = transpose(corr_ab_T, (0, 2, 1))
        a_warp = bmm(a_raw.reshape(n, raw_c, h * w), corr_ba_T)  # n*HW*1
        a_warp = a_warp.reshape(n, raw_c, h, w)
        return corr_ab_T, corr_ba_T, a_warp, b_warp

    def construct(self, fa, fb, a_raw, b_raw):
        fa = self.fa_conv(fa)
        fb = self.fb_conv(fb)
        X, Y, a_warp, b_warp = self.warp(fa, fb, a_raw, b_raw, self.softmax_alpha)
        return X, Y, a_warp, b_warp


####################################################################
# ------------------------------- SSCFT----------------------------
####################################################################
class Transformer(nn.Cell):
    def __init__(self):
        super(Transformer, self).__init__()
        self.fusion = FeatureFusion()
        self.atte = SymmetryAttention()

    def construct(self, x_c, y_c, x_s, y_s, x_m, y_m):
        x_f = self.fusion(x_c, x_s)
        y_f = self.fusion(y_c, y_s)
        attention_x, attention_y, x_m_warp, y_m_warp = self.atte.construct(x_f, y_f, x_m, y_m)
        return attention_x, attention_y, x_m_warp, y_m_warp

####################################################################
# -------------------------- Decoder --------------------------
####################################################################

class Decoder(nn.Cell):
    def __init__(self, output_dim=3, ngf=64):
        super(Decoder, self).__init__()

        self.SPADE1 = SPADEResnetBlock(ngf * 4, ngf * 4, ngf * 4)
        self.SPADE2 = SPADEResnetBlock(ngf * 4, ngf * 2, ngf * 4)
        self.SPADE3 = SPADEResnetBlock(ngf * 2, ngf * 1, ngf * 4)
        self.img_conv = ConvNormReLU(ngf * 1, output_dim, kernel_size=7, stride=1, padding=3, norm_mode='None',
                                    use_relu=False)
        self.tanh = nn.Tanh()

    def construct(self, x, y):
        # content=x[-1]
        # makeup=y
        _,c,h,w=x[-1].shape
        up_1 = ops.ResizeBilinear(size=(h*2,w*2))
        up_2 = ops.ResizeBilinear(size=(h * 4, w * 4))
        out = self.SPADE1(x[-1],y)
        out = up_1(out)
        out = self.SPADE2(out, y)
        out = up_2(out)
        out = self.SPADE3(out, y)
        out= self.img_conv(out)
        out = self.tanh(out)
        return out



####################################################################
# ------------------------------ SPADE -----------------------------
####################################################################

class SPADEResnetBlock(nn.Cell):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, pad_mode='same')
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, pad_mode='same')
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, pad_mode='same', has_bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)
        self.actvn = nn.LeakyReLU(0.2)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def construct(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s


class SPADE(nn.Cell):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = nn.SequentialCell(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, pad_mode='same'),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, pad_mode='same')
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, pad_mode='same')

    def construct(self, x, segmap):
        nearest = ops.ResizeNearestNeighbor((x.shape[2], x.shape[3]))
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map

        segmap = nearest(segmap)
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

class ConvNormReLU(nn.Cell):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 alpha=0.2,
                 norm_mode='instance',
                 pad_mode='REFLECT',
                 use_relu=True,
                 padding=None,
                 has_bias=True):
        super(ConvNormReLU, self).__init__()
        layers = []
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias,
                             padding=padding)
            layers.append(conv)
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers.append(pad)
            layers.append(conv)
        if norm_mode == 'instance':
            norm = nn.InstanceNorm2d(out_planes, affine=False)
            layers.append(norm)
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.model(x)
        return output

####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
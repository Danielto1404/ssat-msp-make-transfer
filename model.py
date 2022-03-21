import network
from network import init_weights
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context


####################################################################
# -------------------------- model --------------------------
####################################################################

class SSAT_D_non_makeup(nn.Cell):
    def __init__(self, opts):
        super(SSAT_D_non_makeup, self).__init__()
        self.dis_non_makeup = None
        if opts.dis_scale > 1:
            self.dis_non_makeup = network.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis_non_makeup = network.Dis(opts.input_dim, norm=opts.dis_norm)

    def construct(self, x):
        return self.dis_non_makeup(x)


class SSAT_D_makeup(nn.Cell):
    def __init__(self, opts):
        super(SSAT_D_makeup, self).__init__()
        self.dis_makeup = None
        if opts.dis_scale > 1:
            self.dis_makeup = network.MultiScaleDis(opts.input_dim, n_scale=opts.dis_scale, norm=opts.dis_norm)
        else:
            self.dis_makeup = network.Dis(opts.input_dim, norm=opts.dis_norm)

    def construct(self, x):
        return self.dis_makeup(x)


class SSAT_G(nn.Cell):
    def __init__(self, opts):
        super(SSAT_G, self).__init__()
        self.opts = opts
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim
        # encoders
        self.enc_content = network.E_content(opts.input_dim)
        self.enc_makeup = network.E_makeup(opts.input_dim)
        self.enc_semantic = network.E_semantic(opts.semantic_dim)
        # FF and SSCFT
        self.transformer = network.Transformer()
        # generator
        self.gen = network.Decoder(opts.output_dim)

    def ouput_fake_images(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)

        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer.construct(z_non_makeup_c,
                                                                                      z_makeup_c,
                                                                                      z_non_makeup_s,
                                                                                      z_makeup_s,
                                                                                      z_non_makeup_a,
                                                                                      z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen(z_makeup_c, z_non_makeup_a_warp)
        return z_transfer, z_removal

    def construct(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse):
        # first transfer and removal
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)

        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer.construct(z_non_makeup_c,
                                                                                      z_makeup_c,
                                                                                      z_non_makeup_s,
                                                                                      z_makeup_s,
                                                                                      z_non_makeup_a,
                                                                                      z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen.construct(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen.construct(z_makeup_c, z_non_makeup_a_warp)

        # rec
        z_rec_non_makeup = self.gen(z_non_makeup_c, z_non_makeup_a)
        z_rec_makeup = self.gen(z_makeup_c, z_makeup_a)

        # second transfer and removal
        z_transfer_c = self.enc_content(z_transfer)
        z_transfer_a = self.enc_makeup(z_transfer)

        z_removal_c = self.enc_content(z_removal)
        z_removal_a = self.enc_makeup(z_removal)
        # warp makeup style
        mapX2, mapY2, z_transfer_a_warp, z_removal_a_warp = self.transformer(z_transfer_c, z_removal_c, z_non_makeup_s,
                                                                             z_makeup_s, z_transfer_a, z_removal_a)

        # makeup transfer and removal
        z_cycle_non_makeup = self.gen(z_transfer_c, z_removal_a_warp)
        z_cycle_makeup = self.gen(z_removal_c, z_transfer_a_warp)
        return z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY

    def output(self, non_makeup, makeup, non_makeup_parse, makeup_parse):
        # first transfer and removal
        z_non_makeup_c = self.enc_content(non_makeup)
        z_non_makeup_s = self.enc_semantic(non_makeup_parse)
        z_non_makeup_a = self.enc_makeup(non_makeup)

        z_makeup_c = self.enc_content(makeup)
        z_makeup_s = self.enc_semantic(makeup_parse)
        z_makeup_a = self.enc_makeup(makeup)

        # warp makeup style
        mapX, mapY, z_non_makeup_a_warp, z_makeup_a_warp = self.transformer.construct(z_non_makeup_c,
                                                                                      z_makeup_c,
                                                                                      z_non_makeup_s,
                                                                                      z_makeup_s,
                                                                                      z_non_makeup_a,
                                                                                      z_makeup_a)
        # makeup transfer and removal
        z_transfer = self.gen.construct(z_non_makeup_c, z_makeup_a_warp)
        z_removal = self.gen.construct(z_makeup_c, z_non_makeup_a_warp)

        # rec
        z_rec_non_makeup = self.gen(z_non_makeup_c, z_non_makeup_a)
        z_rec_makeup = self.gen(z_makeup_c, z_makeup_a)

        # second transfer and removal
        z_transfer_c = self.enc_content(z_transfer)
        z_transfer_a = self.enc_makeup(z_transfer)

        z_removal_c = self.enc_content(z_removal)
        z_removal_a = self.enc_makeup(z_removal)
        # warp makeup style
        mapX2, mapY2, z_transfer_a_warp, z_removal_a_warp = self.transformer(z_transfer_c, z_removal_c, z_non_makeup_s,
                                                                             z_makeup_s, z_transfer_a, z_removal_a)

        # makeup transfer and removal
        z_cycle_non_makeup = self.gen(z_transfer_c, z_removal_a_warp)
        z_cycle_makeup = self.gen(z_removal_c, z_transfer_a_warp)
        return z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY


####################################################################
# -------------------------- get_model --------------------------
####################################################################
def get_generator(opts):
    """Return generator by args."""
    net = SSAT_G(opts)
    init_weights(net, opts.init_type, opts.init_gain)
    return net


def get_dis_non_makeup(opts):
    """Return discriminator by args."""
    net = SSAT_D_non_makeup(opts)
    init_weights(net, opts.init_type, opts.init_gain)
    return net


def get_dis_makeup(opts):
    """Return discriminator by args."""
    net = SSAT_D_makeup(opts)
    init_weights(net, opts.init_type, opts.init_gain)
    return net


####################################################################
# --------------------------train model --------------------------
####################################################################

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse):
        lg,z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = self.network.construct(non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse)
        return lg

class TrainOneStepG(nn.Cell):
    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.G.dis_non_makeup.set_grad(False)
        self.G.dis_non_makeup.set_train(False)
        self.G.dis_makeup.set_grad(False)
        self.G.dis_makeup.set_train(False)
        self.net = WithLossCell(G)
        # self.net.add_flags(defer_inline=True)

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")

    def construct(self, non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse):
        weights = self.weights
        lg,z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY=self.G.construct(non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse)
        #print(lg)
        sens = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.net, weights)(non_makeup, makeup, transfer, removal, non_makeup_parse, makeup_parse, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        self.optimizer(grads_g)
        return lg,z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY


class TrainOneStepD(nn.Cell):
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__()
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(D.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None

    def construct(self, non_makeup, makeup, z_transfer, z_removal):
        weights = self.weights
        ld = self.D(non_makeup, makeup, z_transfer, z_removal)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.D, weights)(non_makeup, makeup, z_transfer, z_removal, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        self.optimizer(grads_d)
        return ld

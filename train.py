# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import mindspore.nn as nn
from reporter import Reporter
from mindspore.context import ParallelMode
from option import get_opts
from mindspore import context
from tools import get_lr
from dataset import create_dataset
from mindspore.communication.management import init, get_rank
from loss import SAATDLoss, SAATGLoss
from model import get_generator, get_dis_non_makeup, get_dis_makeup, TrainOneStepG, TrainOneStepD
import time
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train():
    """Train function."""
    opts = get_opts()
    # context.set_context(mode=context.GRAPH_MODE, device_target=opts.platform, device_id=opts.device_id,
    #                     save_graphs=False)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=opts.platform, device_id=opts.device_id,
                        save_graphs=False)

    ds = create_dataset(opts)
    data_loader = ds.create_dict_iterator()

    # model G and D
    G = get_generator(opts)
    D_non_makeup = get_dis_non_makeup(opts)
    D_makeup = get_dis_makeup(opts)

    # loss G and D
    loss_G = SAATGLoss(opts, G, D_non_makeup, D_makeup)
    loss_D = SAATDLoss(opts, D_non_makeup, D_makeup)

    # optimizer G and D
    optimizer_G = nn.Adam(G.trainable_params(), get_lr(opts), beta1=opts.beta1, beta2=opts.beta2, weight_decay=0.0001)
    optimizer_D = nn.Adam(D_non_makeup.trainable_params() + D_makeup.trainable_params(), get_lr(opts), beta1=opts.beta1,
                          beta2=opts.beta2,
                          weight_decay=0.0001)

    # TrainOneStep G and D
    train_G = TrainOneStepG(loss_G, optimizer_G)
    train_D = TrainOneStepD(loss_D, optimizer_D)

    reporter = Reporter(opts)
    reporter.info('==========start training===============')

    for _ in range(opts.max_epoch):
        reporter.epoch_start()

        for data in data_loader:

            non_makeup = data['non_makeup']
            makeup = data['makeup']
            transfer_g = data['transfer']
            removal_g = data['removal']
            non_makeup_parse = data['non_makeup_parse']
            makeup_parse = data['makeup_parse']
            # print('non_makeup size:',non_makeup.shape)
            g_loss,z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = train_G.construct(non_makeup, makeup, transfer_g, removal_g, non_makeup_parse, makeup_parse)
            # z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = G.construct(
            #     non_makeup, makeup, transfer_g, removal_g, non_makeup_parse, makeup_parse)
            d_loss = train_D(non_makeup, makeup, z_transfer, z_removal)
            print("G_loss:", g_loss.asnumpy(), 'D_loss:', d_loss.asnumpy())

            reporter.step_end(g_loss, d_loss)
            reporter.visualizer(non_makeup, makeup, mapX, mapY, z_transfer, z_removal, transfer_g, removal_g,
                                z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup)

        reporter.epoch_end(train_G)

    reporter.info('==========end training===============')


if __name__ == "__main__":
    train()

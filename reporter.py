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

"""Reporter class."""

import logging
import os
import time
from datetime import datetime

import mindspore.ops as ops
from mindspore.train.serialization import save_checkpoint

from tools import save_image


class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.
    Args:
        args (class): Option class.
    """

    def __init__(self, args):
        super(Reporter, self).__init__("SSAT")
        self.log_dir = os.path.join(args.outputs_dir, 'log')
        self.imgs_dir = os.path.join(args.outputs_dir, "imgs")
        self.ckpts_dir = os.path.join(args.outputs_dir, "ckpt")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        self.save_imgs = args.save_imgs
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.addHandler(console)
        # file handler
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_.log'
        self.log_fn = os.path.join(self.log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.save_args(args)
        self.step = 0
        self.epoch = 0
        self.dataset_size = args.dataset_size // args.device_num
        self.device_num = args.device_num
        self.print_iter = args.print_iter
        self.G_loss = []
        self.D_loss = []

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info('--> %s: %s', key, args_dict[key])
        self.info('')

    def epoch_start(self):
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.G_loss = []
        self.D_loss = []

    def step_end(self, res_G, res_D):
        """print log when step end."""
        self.step += 1
        loss_D = float(res_D.asnumpy())
        loss_G = float(res_G.asnumpy())
        self.G_loss.append(loss_G)
        self.D_loss.append(loss_D)
        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            losses = "G_loss: {:.2f}, D_loss:{:.2f}".format(loss_G, loss_D)
            self.info("Epoch[{}] [{}/{}] step cost: {:.2f} ms, {}".format(
                self.epoch, self.step, self.dataset_size, step_cost, losses))
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """print log and save checkpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        per_step_time = epoch_cost / self.dataset_size
        mean_loss_G = sum(self.G_loss) / self.dataset_size
        mean_loss_D = sum(self.D_loss) / self.dataset_size
        self.info("Epoch [{}] total cost: {:.2f} ms, per step: {:.2f} ms, G_loss: {:.2f}, D_loss: {:.2f}".format(
            self.epoch, epoch_cost, per_step_time, mean_loss_G, mean_loss_D))

        if self.epoch % self.save_checkpoint_epochs == 0:
            save_checkpoint(net.G.gen, os.path.join(self.ckpts_dir, f"SSAT_G_{self.epoch}.ckpt"))
            # save_checkpoint(net.G.dis_non_makeup, os.path.join(self.ckpts_dir, f"SSAT_D_non_makeup_{self.epoch}.ckpt"))
            # save_checkpoint(net.G.dis_makeup, os.path.join(self.ckpts_dir, f"SSAT_D_makeup_{self.epoch}.ckpt"))

    def visualizer(self, non_makeup, makeup, mapX, mapY, z_transfer, z_removal, transfer_g, removal_g,
                   z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup):
        if self.save_imgs and self.step % self.dataset_size == 0:
            _, C, H, W = non_makeup.shape
            concat_2 = ops.Concat(axis=2)
            concat_3 = ops.Concat(axis=3)
            bmm = ops.BatchMatMul()
            nearest_256 = ops.ResizeNearestNeighbor((H, W))
            nearest_64 = ops.ResizeNearestNeighbor((H // 4, W // 4))

            non_makeup_down = nearest_64(non_makeup)
            n, c, h, w = non_makeup_down.shape
            non_makeup_down_warp = bmm(non_makeup_down.reshape(n, c, h * w), mapY)  # n*HW*1
            non_makeup_down_warp = non_makeup_down_warp.reshape(n, c, h, w)
            non_makeup_warp = nearest_256(non_makeup_down_warp)

            makeup_down = nearest_64(makeup)
            n, c, h, w = makeup_down.shape
            makeup_down_warp = bmm(makeup_down.reshape(n, c, h * w), mapX)  # n*HW*1
            makeup_down_warp = makeup_down_warp.reshape(n, c, h, w)
            makeup_warp = nearest_256(makeup_down_warp)

            row_1 = concat_3((non_makeup, makeup_warp, transfer_g, z_transfer, z_rec_non_makeup, z_cycle_non_makeup))
            row_2 = concat_3((makeup, non_makeup_warp, removal_g, z_removal, z_rec_makeup, z_cycle_makeup))
            result = concat_2((row_1, row_2))
            save_image(result, os.path.join(self.imgs_dir, f"{self.epoch}_result.jpg"))

    def start_predict(self, direction):
        self.predict_start_time = time.time()
        self.direction = direction
        self.info('==========start predict %s===============', self.direction)

    def end_predict(self):
        cost = (time.time() - self.predict_start_time) * 1000
        per_step_cost = cost / self.dataset_size
        self.info('total {} imgs cost {:.2f} ms, per img cost {:.2f}'.format(self.dataset_size, cost, per_step_cost))
        self.info('==========end predict %s===============\n', self.direction)

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
import mindspore.ops as ops
from reporter import Reporter
from option import get_opts
from mindspore import context
from tools import get_lr,load_ckpt,save_image
from dataset import create_dataset
from loss import SAATDLoss, SAATGLoss
from model import get_generator, get_dis_non_makeup, get_dis_makeup, TrainOneStepG, TrainOneStepD
import time
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def eval():
    """Train function."""
    opts = get_opts()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=opts.platform, device_id=opts.device_id,
                        save_graphs=False)

    opts.phase='test'
    opts.flip=False
    opts.resize_size=opts.crop_size
    ds = create_dataset(opts)
    data_loader = ds.create_dict_iterator()

    # model G and D
    G = get_generator(opts)
    opts.resume='./outputs/ckpt/SSAT_G_1000.ckpt'
    load_ckpt(opts, G)
    imgs_out = os.path.join(opts.outputs_dir, "predict")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    count=0
    for data in data_loader:
        count+=1
        non_makeup = data['non_makeup']
        makeup = data['makeup']
        non_makeup_parse = data['non_makeup_parse']
        makeup_parse = data['makeup_parse']
        # print('non_makeup size:',non_makeup.shape)
        z_transfer, z_removal, z_rec_non_makeup, z_rec_makeup, z_cycle_non_makeup, z_cycle_makeup, mapX, mapY = G.output(non_makeup, makeup, non_makeup_parse, makeup_parse)

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

        row_1 = concat_3((non_makeup, makeup_warp, z_transfer, z_rec_non_makeup, z_cycle_non_makeup))
        row_2 = concat_3((makeup, non_makeup_warp, z_removal, z_rec_makeup, z_cycle_makeup))
        result = concat_2((row_1, row_2))
        save_image(result, os.path.join(imgs_out, f"{count}_result.jpg"))


if __name__ == "__main__":
    eval()

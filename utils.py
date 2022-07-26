# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import paddle
import paddle.distributed
import glob
import shutil
from pgl.utils.logger import log

import joblib


def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def save_model_gru_base(output_path, model, steps=None, valid_info=None, turbine_id=0):
    if paddle.distributed.get_rank() == 0:
        output_dir = output_path

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        paddle.save(model.state_dict(),
                    os.path.join(output_dir, "model_%s" % str(turbine_id)))

        if valid_info is not None:
            if turbine_id == 0:
                with open(output_dir + '/valid_info.txt', 'w') as f:
                    f.write(str(turbine_id) + '-' + str(steps) + '-score-'+str(valid_info['score']))
            else:
                with open(output_dir + '/valid_info.txt', 'a+') as f:
                    f.write('\n')
                    f.write(str(turbine_id) + '-' + str(steps) + '-score-'+str(valid_info['score']))


def save_model_xgboost(output_path, model, valid_info, turbine_id):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    joblib.dump(model, output_path + 'xgb-%s.dat' % str(turbine_id))


def load_model_xgboost(model_path):
    model = joblib.load(model_path + 'xgb.dat')
    return model

def save_model(output_path,
               model,
               steps=None,
               valid_info=None,
               seed=0,
               opt=None,
               lr_scheduler=None,
               max_ckpt=2):
    if paddle.distributed.get_rank() == 0:
        output_dir = os.path.join(output_path, "model")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        paddle.save(model.state_dict(),
                    os.path.join(output_dir, "ckpt.pdparams-seed%s" % str(seed)))
        if valid_info is not None:
            paddle.save(valid_info, os.path.join(output_dir, 'score-'+str(valid_info['score'])))
        if steps is not None:
            paddle.save({
                "global_step": steps
            }, os.path.join(output_dir, "step-%s" %str(steps)))
        if opt is not None:
            paddle.save(opt.state_dict(),
                        os.path.join(output_dir, "opt.pdparams"))
        if lr_scheduler is not None:
            paddle.save(lr_scheduler.state_dict(),
                        os.path.join(output_dir, "lr_scheduler.pdparams"))
        log.info("save model %s at step %s" % (output_dir, steps))

        # ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
        # if len(ckpt_paths) > max_ckpt:
        #
        #     def version(x):
        #         x = int(x.split("_")[-1])
        #         return x
        #
        #     rm_ckpt_paths = sorted(
        #         ckpt_paths, key=version, reverse=True)[max_ckpt:]
        #     for ckpt_dir in rm_ckpt_paths:
        #         if os.path.exists(ckpt_dir):
        #             shutil.rmtree(ckpt_dir)


def load_model(output_path, model, opt=None, lr_scheduler=None):
    def version(x):
        x = int(x.split("_")[-1])
        return x
    ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
    print(ckpt_paths)

    steps = 0
    if len(ckpt_paths) > 0:
        output_dir = sorted(ckpt_paths, key=version, reverse=True)[0]

        model_state_dict = paddle.load(
            os.path.join(output_dir, "ckpt.pdparams"))

        model.set_state_dict(model_state_dict)

        log.info("load model from  %s" % output_dir)

        if opt is not None and os.path.exists(
                os.path.join(output_dir, "opt.pdparams")):
            opt_state_dict = paddle.load(
                os.path.join(output_dir, "opt.pdparams"))
            opt.set_state_dict(opt_state_dict)
            log.info("restore optimizer")

        if lr_scheduler is not None and os.path.exists(
                os.path.join(output_dir, "lr_scheduler.pdparams")):
            lr_scheduler_state_dict = paddle.load(
                os.path.join(output_dir, "lr_scheduler.pdparams"))
            lr_scheduler.set_state_dict(lr_scheduler_state_dict)
            log.info("restore lr_scheduler")

        if os.path.exists(os.path.join(output_dir, "lr_scheduler.pdparams")):
            steps = paddle.load(os.path.join(output_dir, "step"))[
                "global_step"]
            log.info("restore steps")
    else:
        log.info("no model is loaded!")
        raise ValueError
    return steps


def load_model_unique(output_path, model, model_name):
    model_state_dict = paddle.load(
        os.path.join(output_path, model_name))
    model.set_state_dict(model_state_dict)
    log.info("    + load model from  %s %s" % (output_path, model_name))
    return True
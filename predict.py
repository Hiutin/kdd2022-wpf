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
import time
from common import Experiment

import glob
import argparse
import paddle
import paddle.nn.functional as F
import tqdm
import yaml
import numpy as np

import pgl
from pgl.utils.logger import log
from paddle.io import DataLoader
import random

import loss as loss_factory
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from wpf_model import WPFModel
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model, load_model_unique
import matplotlib.pyplot as plt

from model import BaselineGruModel

from wpf_model_bn_nn import WPFModelBNNN
from wpf_model_no_gat import WPFModelNOGAT
from wpf_model_short_input import WPFModelShort

@paddle.no_grad()
def forecast(envs):
    def model_filter(f):
        if f[:13] == "ckpt.pdparams":
            return True
        else:
            return False

    # load train data for data scale
    train_data = PGL4WPFDataset(
        envs['data_path'],
        filename=envs['filename'],
        size=[envs['input_len'], envs['output_len']],
        flag='train',
        total_days=envs['total_size'],
        train_days=envs['train_size'],
        val_days=envs['val_size'],
        test_days=envs['total_size']-envs['train_size']-envs['val_size'])

    # data scale
    data_mean = paddle.to_tensor(train_data.data_mean, dtype="float32")
    data_scale = paddle.to_tensor(train_data.data_scale, dtype="float32")

    graph = train_data.graph
    graph = graph.tensor()

    # load test x
    x_file = envs['path_to_test_x']
    test_x_ds = TestPGL4WPFDataset(filename=x_file)
    test_x = paddle.to_tensor(test_x_ds.get_data()[:, :, -envs['input_len']:, :], dtype="float32")
    test_x = test_x[:, :, :, 2:]
    test_x = (test_x - data_mean) / data_scale  # test_x: [batch_size, id_len, input_len, var_len]

    assert envs['ensemble'] in [0, 1, 2, 3]

    path_model_graph = 'model/'
    path_model_gru = 'model_gru2/'
    path_model_gru_new = 'model_gru/'
    path_model_graph_bn_nn = 'model_bn/'
    path_model_graph_no_gat = 'model_nogat/'
    path_model_graph_short = 'model_short/'

    if envs['ensemble'] == 0:
        # load gru model and predict
        log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru))

        pred_all_turbine = []
        for i in range(envs["capacity"]):
            model = BaselineGruModel(envs)
            path_to_model = os.path.join(envs["checkpoints"], path_model_gru + 'model_{}'.format(str(i)))
            model.set_state_dict(paddle.load(path_to_model))
            pred_y = model(test_x[:, i, :, :], data_mean, data_scale) # send i-th input to model

            pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]

            pred_all_turbine.append(pred_y)
        pred_all_turbine = np.array(pred_all_turbine)
        pred_all_turbine = paddle.to_tensor(pred_all_turbine)
        pred_all_turbine = paddle.transpose(pred_all_turbine, [1, 0, 2, 3]) # [batch_size, id_len, output_len, var_len]

        # #load gru model new and predict
        # log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru))
        # pred_all_turbine_new2 = []
        # for i in range(envs["capacity"]):
        #     model = BaselineGruModel(envs)
        #     path_to_model = os.path.join(envs["checkpoints"], path_model_gru + 'model_{}'.format(str(i)))
        #     model.set_state_dict(paddle.load(path_to_model))
        #     pred_y = model(test_x[:, i, :, :], data_mean, data_scale)  # send i-th input to model
        #
        #     pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]
        #
        #     pred_all_turbine_new2.append(pred_y)
        # pred_all_turbine_new2 = np.array(pred_all_turbine_new2)
        # pred_all_turbine_new2 = paddle.to_tensor(pred_all_turbine_new2)
        # pred_all_turbine_new2 = paddle.transpose(pred_all_turbine_new2, [1, 0, 2, 3])  # [batch_size, id_len, output_len, var_len]

        # #load gru model new and predict
        # log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru_new))
        # pred_all_turbine_new = []
        # for i in range(envs["capacity"]):
        #     model = BaselineGruModel(envs)
        #     path_to_model = os.path.join(envs["checkpoints"], path_model_gru_new + 'model_{}'.format(str(i)))
        #     model.set_state_dict(paddle.load(path_to_model))
        #     pred_y = model(test_x[:, i, :, :], data_mean, data_scale)  # send i-th input to model
        #
        #     pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]
        #
        #     pred_all_turbine_new.append(pred_y)
        # pred_all_turbine_new = np.array(pred_all_turbine_new)
        # pred_all_turbine_new = paddle.to_tensor(pred_all_turbine_new)
        # pred_all_turbine_new = paddle.transpose(pred_all_turbine_new, [1, 0, 2, 3])  # [batch_size, id_len, output_len, var_len]

        # load graph-autoformer and predict
        log.info("load graph-autoformer model and predict by %s", os.path.join(envs["checkpoints"], path_model_graph))
        path_to_model = os.path.join(envs["checkpoints"], path_model_graph)
        models = list(filter(model_filter, os.listdir(path_to_model)))
        pred_all_models = []
        for m in models:
            info = m.split('-')
            seed_num = int(info[1][4:])

            envs['seed_num'] = seed_num
            model = WPFModel(config=envs)
            load_model_unique(path_to_model, model, m)
            model.eval()

            pred_y = model(test_x, data_mean, data_scale, graph)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.numpy(), -1)

            pred_all_models.append(pred_y)

        all_model_pred = np.array(pred_all_models) # [id_model, batch_size, id_len, output_len, var_len]
        all_model_pred_average = paddle.to_tensor(np.mean(all_model_pred, axis=0))

        # # load WPFModelBNNN and predict
        # log.info("load graph-autoformer model and predict by %s", os.path.join(envs["checkpoints"], path_model_graph_bn_nn))
        # path_to_model = os.path.join(envs["checkpoints"], path_model_graph_bn_nn)
        # models = list(filter(model_filter, os.listdir(path_to_model)))
        # pred_all_models_BNNN = []
        # for m in models:
        #     info = m.split('-')
        #     seed_num = int(info[1][4:])
        #
        #     envs['seed_num'] = seed_num
        #     model = WPFModelBNNN(config=envs)
        #     load_model_unique(path_to_model, model, m)
        #     model.eval()
        #
        #     pred_y = model(test_x, data_mean, data_scale, graph)
        #     pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        #
        #     pred_y = np.expand_dims(pred_y.numpy(), -1)
        #
        #     pred_all_models_BNNN.append(pred_y)
        #
        # all_model_pred_BNNN = np.array(pred_all_models_BNNN)  # [id_model, batch_size, id_len, output_len, var_len]
        # all_model_pred_average_BNNN = paddle.to_tensor(np.mean(all_model_pred_BNNN, axis=0))

        # # load WPFModelNOGAT and predict
        # log.info("load graph-autoformer model and predict by %s",
        #          os.path.join(envs["checkpoints"], path_model_graph_short))
        # path_to_model = os.path.join(envs["checkpoints"], path_model_graph_short)
        # models = list(filter(model_filter, os.listdir(path_to_model)))
        # pred_all_models_NOGAT = []
        # for m in models:
        #     info = m.split('-')
        #     seed_num = int(info[1][4:])
        #
        #     envs['seed_num'] = seed_num
        #     model = WPFModelShort(config=envs)
        #     load_model_unique(path_to_model, model, m)
        #     model.eval()
        #
        #     pred_y = model(test_x, data_mean, data_scale, graph)
        #     pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        #
        #     pred_y = np.expand_dims(pred_y.numpy(), -1)
        #
        #     pred_all_models_NOGAT.append(pred_y)
        #
        # all_model_pred_NOGAT = np.array(pred_all_models_NOGAT)  # [id_model, batch_size, id_len, output_len, var_len]
        # all_model_pred_average_NOGAT = paddle.to_tensor(np.mean(all_model_pred_NOGAT, axis=0))

        # # load WPFModelShort and predict
        log.info("load graph-autoformer model and predict by %s",
                 os.path.join(envs["checkpoints"], path_model_graph_short))
        path_to_model = os.path.join(envs["checkpoints"], path_model_graph_short)
        models = list(filter(model_filter, os.listdir(path_to_model)))
        pred_all_models_short = []
        for m in models:
            info = m.split('-')
            seed_num = int(info[1][4:])

            envs['seed_num'] = seed_num
            model = WPFModelShort(config=envs)
            load_model_unique(path_to_model, model, m)
            model.eval()

            pred_y = model(test_x, data_mean, data_scale, graph)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.numpy(), -1)

            pred_all_models_short.append(pred_y)

        all_model_pred_short = np.array(pred_all_models_short)  # [id_model, batch_size, id_len, output_len, var_len]
        all_model_pred_average_short = paddle.to_tensor(np.mean(all_model_pred_short, axis=0))

        # average of gru and graph-autoformer

        # output_all = paddle.concat([pred_all_turbine, pred_all_turbine_new, pred_all_turbine_new2, all_model_pred_average, all_model_pred_average_BNNN, all_model_pred_average_NOGAT, all_model_pred_average_short], 3)
        # output_all = paddle.concat([pred_all_turbine, pred_all_turbine_new, all_model_pred_average, all_model_pred_average_short], 3)
        output_all = paddle.concat([pred_all_turbine, all_model_pred_average, all_model_pred_average_short], 3)
        # output_all = paddle.concat([pred_all_turbine, pred_all_turbine_new, all_model_pred_average, all_model_pred_average_BNNN, all_model_pred_average_short], 3)
        output = paddle.mean(output_all, 3)
        output = np.expand_dims(output.numpy(), -1)
        output = output[0, :, :, :]
        return output

    if envs['ensemble'] == 1:
        # load gru model and predict
        log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru))
        pred_all_turbine = []
        for i in range(envs["capacity"]):
            model = BaselineGruModel(envs)
            path_to_model = os.path.join(envs["checkpoints"], path_model_gru + 'model_{}'.format(str(i)))
            model.set_state_dict(paddle.load(path_to_model))
            pred_y = model(test_x[:, i, :, :], data_mean, data_scale)

            pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]

            pred_all_turbine.append(pred_y)
        pred_all_turbine = np.array(pred_all_turbine)
        pred_all_turbine = paddle.to_tensor(pred_all_turbine)
        output = paddle.transpose(pred_all_turbine, [1, 0, 2, 3])  # [batch_size, id_len, output_len, var_len]

        output = output.numpy()[0, :, :, :]
        return output

    if envs['ensemble'] == 2:
        # load graph-autoformer and predict
        log.info("load graph-autoformer model and predict by %s", os.path.join(envs["checkpoints"], path_model_graph))
        path_to_model = os.path.join(envs["checkpoints"], path_model_graph)
        models = list(filter(model_filter, os.listdir(path_to_model)))
        pred_all_models = []
        for m in models:
            info = m.split('-')
            seed_num = int(info[1][4:])

            envs['seed_num'] = seed_num
            model = WPFModel(config=envs)
            load_model_unique(path_to_model, model, m)
            model.eval()

            pred_y = model(test_x, data_mean, data_scale, graph)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.numpy(), -1)

            pred_all_models.append(pred_y)

        all_model_pred = np.array(pred_all_models)  # [id_model, batch_size, id_len, output_len, var_len]
        output = np.mean(all_model_pred, axis=0)
        output = output[0, :, :, :]
        return output

    if envs['ensemble'] == 3:
        # load gru model and predict
        log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru))

        pred_all_turbine = []
        for i in range(envs["capacity"]):
            model = BaselineGruModel(envs)
            path_to_model = os.path.join(envs["checkpoints"], path_model_gru + 'model_{}'.format(str(i)))
            model.set_state_dict(paddle.load(path_to_model))
            pred_y = model(test_x[:, i, :, :], data_mean, data_scale) # send i-th input to model

            pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]

            pred_all_turbine.append(pred_y)
        pred_all_turbine = np.array(pred_all_turbine)
        pred_all_turbine = paddle.to_tensor(pred_all_turbine)
        pred_all_turbine = paddle.transpose(pred_all_turbine, [1, 0, 2, 3]) # [batch_size, id_len, output_len, var_len]

        # # load graph-autoformer and predict
        # log.info("load graph-autoformer model and predict by %s", os.path.join(envs["checkpoints"], path_model_graph))
        # path_to_model = os.path.join(envs["checkpoints"], path_model_graph)
        # models = list(filter(model_filter, os.listdir(path_to_model)))
        # pred_all_models = []
        # for m in models:
        #     info = m.split('-')
        #     seed_num = int(info[1][4:])
        #
        #     envs['seed_num'] = seed_num
        #     model = WPFModel(config=envs)
        #     load_model_unique(path_to_model, model, m)
        #     model.eval()
        #
        #     pred_y = model(test_x, data_mean, data_scale, graph)
        #     pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])
        #
        #     pred_y = np.expand_dims(pred_y.numpy(), -1)
        #
        #     pred_all_models.append(pred_y)
        #
        # all_model_pred = np.array(pred_all_models) # [id_model, batch_size, id_len, output_len, var_len]
        # all_model_pred_average = paddle.to_tensor(np.mean(all_model_pred, axis=0))

        # load gru model new and predict
        log.info("load GRU model and predict by %s", os.path.join(envs["checkpoints"], path_model_gru_new))
        pred_all_turbine_new = []
        for i in range(envs["capacity"]):
            model = BaselineGruModel(envs)
            path_to_model = os.path.join(envs["checkpoints"], path_model_gru_new + 'model_{}'.format(str(i)))
            model.set_state_dict(paddle.load(path_to_model))
            pred_y = model(test_x[:, i, :, :], data_mean, data_scale)  # send i-th input to model

            pred_y = pred_y * data_scale[:, i, :, -1] + data_mean[:, i, :, -1]

            pred_all_turbine_new.append(pred_y)
        pred_all_turbine_new = np.array(pred_all_turbine_new)
        pred_all_turbine_new = paddle.to_tensor(pred_all_turbine_new)
        pred_all_turbine_new = paddle.transpose(pred_all_turbine_new, [1, 0, 2, 3])  # [batch_size, id_len, output_len, var_len]


        # load graph-autoformer and predict
        log.info("load graph-autoformer model and predict by %s", os.path.join(envs["checkpoints"], path_model_graph))
        path_to_model = os.path.join(envs["checkpoints"], path_model_graph)
        models = list(filter(model_filter, os.listdir(path_to_model)))

        pred_all_models = []
        for m in models:
            info = m.split('-')
            seed_num = int(info[1][4:])

            envs['seed_num'] = seed_num
            model = WPFModel(config=envs)
            load_model_unique(path_to_model, model, m)
            model.eval()

            pred_y = model(test_x, data_mean, data_scale, graph)
            pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :, -1])

            pred_y = np.expand_dims(pred_y.numpy(), -1)

            pred_all_models.append(pred_y)

        all_model_pred = np.array(pred_all_models)  # [id_model, batch_size, id_len, output_len, var_len]
        all_model_pred_average = paddle.to_tensor(np.mean(all_model_pred, axis=0))
        # average of gru and graph-autoformer

        output_all = paddle.concat([pred_all_turbine, pred_all_turbine_new, all_model_pred_average], 3)

        output = paddle.mean(output_all, 3)
        output = np.expand_dims(output.numpy(), -1)
        output = output[0, :, :, :]
        return output
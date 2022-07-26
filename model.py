# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Aaron
Date:    2022/05/20
"""
import paddle
import paddle.nn as nn
import pgl


class GruModelBatch(nn.Layer):
    """
    Desc:
        A simple GRU model with batch
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(GruModelBatch, self).__init__()
        self.output_len = settings["output_len"]
        self.hidR = 48
        self.capacity = settings["capacity"]
        self.hidC = int(settings["in_var"] * settings["capacity"])
        self.out_dim = int(settings["out_var"] * settings["capacity"])
        self.dropout = nn.Dropout(settings['gru']["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out_dim)

    def forward(self, batch_x, data_mean, data_scale, graph=None):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            batch_x, data_mean, data_scale
        Returns:
            A tensor
        """

        bz, id_len, input_len, var_len = batch_x.shape

        if graph is not None:
            batch_graph = pgl.Graph.batch([graph] * bz)

        # the first variable is time and the second is weekday

        var_len = var_len - 2

        time_id = batch_x[:, 0, :, 1].astype("int32")
        weekday_id = batch_x[:, 0, :, 0].astype("int32")

        # dimension of input for each turbine is 10
        batch_x = batch_x[:, :, :, 2:]

        batch_x = (batch_x - data_mean) / data_scale

        batch_x = paddle.transpose(batch_x, [0, 2, 1, 3])  # bz, input_len, id_len, var_len
        batch_x = paddle.reshape(batch_x, [bz, input_len, id_len * var_len]) # bz, input_len, id_len*var_len
        x = paddle.zeros([batch_x.shape[0], self.output_len, batch_x.shape[2]])
        x_enc = paddle.concat((batch_x, x), 1)

        # x_enc = paddle.static.nn.batch_norm(x_enc)

        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        dec = paddle.static.nn.batch_norm(dec)
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out_dim:]
        pred_y = paddle.transpose(sample, [0, 2, 1])
        return pred_y  # [batch_size, id_len, output_len]


class BaselineGruModel(nn.Layer):
    """
    Desc:
        A simple GRU model one by one
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out_dim = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out_dim)

    def forward(self, batch_x, data_mean, data_scale):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            batch_x: the tensor should have been normalized.
        Returns:
            A tensor
        """

        x = paddle.zeros([batch_x.shape[0], self.output_len, batch_x.shape[2]])
        x_enc = paddle.concat((batch_x, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out_dim:]

        return sample  # [B, L, D]
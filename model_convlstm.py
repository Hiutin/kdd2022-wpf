# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: ConvLSTM
Authors: Aaron
Date:    2022/05/20
"""
import paddle
import paddle.nn as nn
import pgl


class ConvLSTM2D(nn.Layer):
    """
    Desc:
        A simple ConvLSTM2D
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(ConvLSTM2D, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out_dim = settings["out_var"]
        self.dropout = nn.Dropout(settings['gru']["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out_dim)

    # def forward(self, x_enc):
    #     # type: (paddle.tensor) -> paddle.tensor
    #     """
    #     Desc:
    #         The specific implementation for interface forward
    #     Args:
    #         x_enc:
    #     Returns:
    #         A tensor
    #     """
    #     x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])
    #     x_enc = paddle.concat((x_enc, x), 1)
    #     x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
    #     dec, _ = self.lstm(x_enc)
    #     dec = paddle.transpose(dec, perm=(1, 0, 2))
    #     sample = self.projection(self.dropout(dec))
    #     sample = sample[:, -self.output_len:, -self.out_dim:]
    #     return sample  # [B, L, D]

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

        x = paddle.zeros([batch_x.shape[0], self.output_len, batch_x.shape[2]])
        x_enc = paddle.concat((batch_x, x), 1)
        print(x_enc)
        print(x_enc.shape)

        raise ValueError
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
        output_len = self.output_len

        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out_dim:]
        return sample  # [B, L, D]

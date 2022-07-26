# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import paddle


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """

    settings = {
        "path_to_test_x": "/var/paddle/sub/data/new_x",
        "path_to_test_y": "/var/paddle/sub/data/new_y",
        "data_path": "/var/paddle/data",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size": 213,
        "val_size": 16,
        "total_size": 245,
        "lstm_layer": 2,
        "dropout": 0.05,
        "num_workers": 0,
        "train_epochs": 10,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True,
        "GPU-device": True,

        # new
        "patient": 2,
        "model": {
            "hidden_dims": 128,
            "nhead": 8,
            "dropout": 0.5,
            "encoder_layers": 2,
            "decoder_layers": 1,

        },
        "gru": {
            "dropout": 0.05,
        },
        "epoch_gru_one": 10,
        "patient_gru_one": 2,
        "input_len": 144,
        "output_len": 288,
        "batch_size": 32,
        "train_days": 213,
        "val_days": 16,
        "test_days": 16,
        "total_days": 245,
        "epoch": 10,
        "output_path": "/var/paddle/output",
        "log_per_steps": 100,
        "lr": 0.00005,
        "shared_memory": False,
        "loss": {
            "name": "FilterMSELoss"
        },
        "seed_list": [3],
        "seed_num": 3,
        "ensemble": 0 # 0: gru + graph-autoformer; 1: gru; 2: graph_autoformer
    }
    ###
    # Prepare the GPUs
    if settings['GPU-device']:
        if paddle.device.is_compiled_with_cuda():
            settings["use_gpu"] = True
            paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
        else:
            settings["use_gpu"] = False
            paddle.device.set_device('cpu')
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')
    # print("The experimental settings are: \n{}".format(str(settings)))
    return settings

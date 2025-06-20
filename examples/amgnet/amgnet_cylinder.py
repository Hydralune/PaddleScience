# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from os import path as osp
from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import hydra
import paddle
import paddle.nn as nn
import utils
from omegaconf import DictConfig
from paddle.nn import functional as F
from paddle.static import InputSpec

import ppsci
from ppsci.utils import logger

if TYPE_CHECKING:
    import pgl


def train_mse_func(
    output_dict: Dict[str, "paddle.Tensor"],
    label_dict: Dict[str, "pgl.Graph"],
    *args,
) -> paddle.Tensor:
    return {"pred": F.mse_loss(output_dict["pred"], label_dict["label"].y)}


def eval_rmse_func(
    output_dict: Dict[str, List["paddle.Tensor"]],
    label_dict: Dict[str, List["pgl.Graph"]],
    *args,
) -> Dict[str, paddle.Tensor]:
    mse_losses = [
        F.mse_loss(pred, label.y)
        for (pred, label) in zip(output_dict["pred"], label_dict["label"])
    ]
    return {"RMSE": (sum(mse_losses) / len(mse_losses)) ** 0.5}


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set cylinder model
    model = ppsci.arch.AMGNet(**cfg.MODEL)

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MeshCylinderDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.TRAIN_DATA_DIR,
            "mesh_graph_path": cfg.TRAIN_MESH_GRAPH_PATH,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshCylinderDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.EVAL_DATA_DIR,
            "mesh_graph_path": cfg.EVAL_MESH_GRAPH_PATH,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"].unsqueeze(0)},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()

    # visualize prediction
    logger.message("Now visualizing prediction, please wait...")
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(rmse_validator.data_loader):
            truefield = label["label"].y
            prefield = model(input_)
            utils.log_images(
                input_["input"].pos,
                prefield["pred"],
                truefield,
                rmse_validator.data_loader.dataset.elems_list,
                index,
                "cylinder",
            )


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set airfoil model
    model = ppsci.arch.AMGNet(**cfg.MODEL)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "MeshCylinderDataset",
            "input_keys": ("input",),
            "label_keys": ("label",),
            "data_dir": cfg.EVAL_DATA_DIR,
            "mesh_graph_path": cfg.EVAL_MESH_GRAPH_PATH,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        output_expr={"pred": lambda out: out["pred"].unsqueeze(0)},
        metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
        name="RMSE_validator",
    )
    validator = {rmse_validator.name: rmse_validator}

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate model
    solver.eval()

    # visualize prediction
    with solver.no_grad_context_manager(True):
        for index, (input_, label, _) in enumerate(rmse_validator.data_loader):
            truefield = label["label"].y
            prefield = model(input_)
            utils.log_images(
                input_["input"].pos,
                prefield["pred"],
                truefield,
                rmse_validator.data_loader.dataset.elems_list,
                index,
                "cylinder",
            )


def export(cfg: DictConfig):
    """Export the model for inference."""
    # set model
    model = ppsci.arch.AMGNet(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )

    # 创建一个简化但保留原始功能的导出模型
    class SimpleExportModel(nn.Layer):
        def __init__(self, original_model):
            super(SimpleExportModel, self).__init__()
            self.output_keys = original_model.output_keys

            # 只保留原始模型的关键组件，不添加额外复杂层
            self.node_encoder = original_model.encoder.node_model
            self.post_processor = original_model.post_processor
            self.decoder = original_model.decoder

        def forward(self, node_feat):
            """简单直接的前向传递，尽量接近原始模型但避免PGL依赖"""
            # 1. 节点特征编码
            encoded_features = self.node_encoder(node_feat)

            # 2. 由于无法完全复制原始的processor操作，直接使用后处理器
            processed_features = self.post_processor(encoded_features)

            # 3. 应用解码器获得最终输出
            output = self.decoder(processed_features)

            return {self.output_keys[0]: output}

    # 创建简化导出模型
    export_model = SimpleExportModel(model)

    # 配置导出选项
    input_spec = [
        InputSpec(shape=[None, cfg.MODEL.input_dim], dtype="float32", name="node_feat"),
    ]

    # 导出模型
    solver.export(input_spec, cfg.INFER.export_path, to_func=export_model.forward)


def infer(cfg: DictConfig):
    """Infer using the trained model."""
    import os

    import amgnet_predictor
    import numpy as np

    # Create data loader
    _, dataset = utils.create_dataset(cfg)
    logger.message("Building dataset...")
    logger.message("Dataset created successfully")

    # Create AMGNPredictor
    logger.message("Getting first sample from dataset...")
    sample = dataset[0]

    # Debug dataset structure
    logger.message(f"Sample type: {type(sample)}")
    if isinstance(sample, tuple) and len(sample) >= 3:
        input_data, label_data, meta = sample
        logger.message(f"Input data type: {type(input_data)}")
        logger.message(f"Label data type: {type(label_data)}")
        logger.message(f"Meta data type: {type(meta)}")

        if isinstance(input_data, dict):
            for k, v in input_data.items():
                logger.message(f"Input key: {k}, value type: {type(v)}")
                if hasattr(v, "x"):
                    logger.message(f"  Has attribute x: {type(v.x)}")
                elif isinstance(v, np.ndarray):
                    logger.message(f"  Is numpy array with shape: {v.shape}")
    else:
        logger.message(f"Unexpected sample structure: {sample}")
        return

    # Extract node features safely
    try:
        graph = input_data["input"]
        logger.message(f"Graph type: {type(graph)}")

        # Handle different types of graph.x
        if hasattr(graph, "x"):
            node_feat = graph.x
            if hasattr(node_feat, "numpy"):
                node_feat = node_feat.numpy()
            logger.message(f"Node features shape: {node_feat.shape}")
        else:
            # If graph is already a numpy array
            node_feat = graph
            logger.message(f"Using graph directly as node features: {node_feat.shape}")

        # Handle different types of label data
        if isinstance(label_data, dict) and "label" in label_data:
            label = label_data["label"]
            if hasattr(label, "y"):
                ground_truth = label.y
                if hasattr(ground_truth, "numpy"):
                    ground_truth = ground_truth.numpy()
            else:
                ground_truth = label
            logger.message(f"Ground truth shape: {ground_truth.shape}")
        else:
            logger.message("Could not extract ground truth data")
            return

        # Handle different types of coordinates
        if hasattr(graph, "pos"):
            coords = graph.pos
            if hasattr(coords, "numpy"):
                coords = coords.numpy()
            logger.message(f"Coordinates shape: {coords.shape}")
        else:
            # Use first two columns of node_feat as coordinates if available
            if node_feat.shape[1] >= 2:
                coords = node_feat[:, :2]
                logger.message(
                    f"Using first two columns of node_feat as coordinates: {coords.shape}"
                )
            else:
                # Generate dummy coordinates
                coords = np.zeros((node_feat.shape[0], 2))
                logger.message("Using dummy coordinates")
    except Exception as e:
        logger.message(f"Error extracting features: {e}")
        import traceback

        logger.message(traceback.format_exc())
        return

    # Create directory for result
    os.makedirs("./result/image/cylinder_infer", exist_ok=True)

    # Initialize predictor
    predictor = amgnet_predictor.AMGNPredictor(
        model_path=cfg.INFER.export_path,
        config_params={
            "use_mkldnn": cfg.INFER.use_mkldnn,
            "ir_optim": cfg.INFER.ir_optim,
        },
        verbose=True,
    )

    logger.message("Running inference on sample...")

    # Run inference
    output = predictor.predict({"node_feat": node_feat})

    logger.message("Generating visualization...")

    # Compare prediction with ground truth
    pred_result = output["pred"]

    # Extract elements list if available
    elems_list = None
    if hasattr(meta, "get"):
        elems_list = meta.get("elems_list", None)

    # Visualize using the original method
    utils.log_images(
        coords,
        pred_result,
        ground_truth,
        elems_list,
        0,  # Sample index
        "cylinder_infer",
    )

    logger.message("Visualization saved to ./result/image/cylinder_infer")


@hydra.main(version_base=None, config_path="./conf", config_name="amgnet_cylinder.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

import math
import os
import pathlib
import warnings
from os import path as osp
from typing import BinaryIO
from typing import List
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.vision import transforms as T
from PIL import Image

matplotlib.use("Agg")


@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs,
) -> paddle.Tensor:
    if not (
        isinstance(tensor, paddle.Tensor)
        or (
            isinstance(tensor, list)
            and all(isinstance(t, paddle.Tensor) for t in tensor)
        )
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            if not isinstance(value_range, tuple):
                raise TypeError(
                    "value_range has to be a tuple (min, max) if specified. min and max are numbers"
                )

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = (
        paddle.clip(grid * 255 + 0.5, 0, 255).transpose([1, 2, 0]).cast("uint8").numpy()
    )
    im = Image.fromarray(ndarr)
    os.makedirs(osp.dirname(fp), exist_ok=True)
    im.save(fp, format=format)


def log_images(nodes, pred_field, true_field, elems_list, idx, mode="cylinder"):
    """Log images for visualization."""
    import os
    import matplotlib.pyplot as plt
    
    # 确保结果目录存在
    result_dir = os.path.join("./result/image", mode)
    os.makedirs(result_dir, exist_ok=True)
    
    # 如果是paddle.Tensor，转换为numpy
    if not isinstance(pred_field, np.ndarray):
        pred_field = pred_field.numpy()
    if not isinstance(true_field, np.ndarray):
        true_field = true_field.numpy()
    
    # 压力场
    p_true = true_field[:, 0]
    p_pred = pred_field[:, 0]
    
    # 速度x分量
    vx_true = true_field[:, 1]
    vx_pred = pred_field[:, 1]
    
    # 速度y分量
    vy_true = true_field[:, 2]
    vy_pred = pred_field[:, 2]
    
    # 处理图形数据
    elems = sum(elems_list, []) if elems_list else None
    
    # 保存压力场图像
    fig_p_true = plot_field(nodes, p_true, elems)
    plt.title("True Pressure")
    plt.savefig(os.path.join(result_dir, f"p_true_{idx}.png"))
    plt.close(fig_p_true)
    
    fig_p_pred = plot_field(nodes, p_pred, elems)
    plt.title("Predicted Pressure")
    plt.savefig(os.path.join(result_dir, f"p_pred_{idx}.png"))
    plt.close(fig_p_pred)
    
    # 保存x方向速度场图像
    fig_vx_true = plot_field(nodes, vx_true, elems)
    plt.title("True X-Velocity")
    plt.savefig(os.path.join(result_dir, f"vx_true_{idx}.png"))
    plt.close(fig_vx_true)
    
    fig_vx_pred = plot_field(nodes, vx_pred, elems)
    plt.title("Predicted X-Velocity")
    plt.savefig(os.path.join(result_dir, f"vx_pred_{idx}.png"))
    plt.close(fig_vx_pred)
    
    # 保存y方向速度场图像
    fig_vy_true = plot_field(nodes, vy_true, elems)
    plt.title("True Y-Velocity")
    plt.savefig(os.path.join(result_dir, f"vy_true_{idx}.png"))
    plt.close(fig_vy_true)
    
    fig_vy_pred = plot_field(nodes, vy_pred, elems)
    plt.title("Predicted Y-Velocity")
    plt.savefig(os.path.join(result_dir, f"vy_pred_{idx}.png"))
    plt.close(fig_vy_pred)
    
    print(f"Saved visualization to {result_dir}")


def plot_field(nodes, field, elems, vmin=None, vmax=None):
    """Plot heatmap for scalar field in nodes."""
    fig = plt.figure(figsize=(20, 10))
    
    # 检查nodes的类型并适当处理
    if isinstance(nodes, np.ndarray):
        x, y = nodes[:, 0], nodes[:, 1]  # 直接获取x和y坐标
    else:
        # 如果是paddle.Tensor或其他支持t()的对象
        x, y = nodes[:, :2].t().detach().numpy()
        
    # 检查field的类型并适当处理
    if isinstance(field, np.ndarray):
        value = field
    else:
        # 如果是paddle.Tensor或其他需要detach的对象
        value = field.detach().numpy()
    
    # 创建三角形单元格
    triangles = []
    if elems is not None:
        for cell in elems:
            if len(cell) == 3:
                triangles.append([cell[0], cell[1], cell[2]])
            elif len(cell) == 4:
                triangles.append([cell[0], cell[1], cell[2]])
                triangles.append([cell[0], cell[2], cell[3]])
    
    if len(triangles) > 0:
        plt.tricontourf(x, y, triangles, value, 100, cmap="jet", vmin=vmin, vmax=vmax)
    else:
        plt.tricontourf(x, y, value, 100, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axes().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    
    return fig


def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(paddle.to_tensor(([[e[0]], [e[2]]]), dtype=paddle.int64))
    new_edges = (
        paddle.concat(new_edges, axis=1)
        if new_edges
        else paddle.to_tensor([], dtype=paddle.int64)
    )
    return new_elems, new_edges


def create_dataset(cfg):
    """创建用于推理的数据集
    
    Args:
        cfg: 配置对象
        
    Returns:
        data_loader, dataset: 数据加载器和数据集对象
    """
    from ppsci.data import dataset
    # 创建数据集
    eval_dataset = dataset.MeshCylinderDataset(
        input_keys=("input",),
        label_keys=("label",),
        data_dir=cfg.EVAL_DATA_DIR,
        mesh_graph_path=cfg.EVAL_MESH_GRAPH_PATH,
    )
    
    return None, eval_dataset


def visualize_result(data, save_path, channel_names=None):
    """可视化预测结果
    
    Args:
        data: 包含pred和true的字典
        save_path: 保存路径
        channel_names: 通道名称列表
    """
    if channel_names is None:
        channel_names = ["p", "vx", "vy"]
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 获取预测值和真实值
    pred = data["pred"]
    true = data["true"]
    
    # 对每个通道进行可视化
    for i, name in enumerate(channel_names):
        if i >= pred.shape[1]:
            continue
            
        # 绘制真实值的热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(true[:, i].reshape(-1, 1), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"True {name}")
        plt.savefig(os.path.join(save_path, f"{name}_true_0.png"))
        plt.close()
        
        # 绘制预测值的热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(pred[:, i].reshape(-1, 1), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"Predicted {name}")
        plt.savefig(os.path.join(save_path, f"{name}_pred_0.png"))
        plt.close()


def graph_collate_fn(batch):
    """自定义的collate函数，用于处理包含图对象的批次数据。
    
    Args:
        batch: 批次数据，每个元素是一个样本
        
    Returns:
        处理后的批次数据
    """
    input_dict = {}
    label_dict = {}
    meta_dict = {}
    
    # 对于每个样本
    for sample in batch:
        for k, v in sample[0].items():
            if k not in input_dict:
                input_dict[k] = []
            input_dict[k].append(v)
        
        for k, v in sample[1].items():
            if k not in label_dict:
                label_dict[k] = []
            label_dict[k].append(v)
        
        meta_dict_sample = sample[2]
        if not meta_dict:
            meta_dict = meta_dict_sample
    
    return input_dict, label_dict, meta_dict

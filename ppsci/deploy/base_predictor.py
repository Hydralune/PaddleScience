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

import os
from typing import Dict

import paddle
import paddle.inference as paddle_infer
from omegaconf import DictConfig

from ppsci.utils import logger


class Predictor:
    """Base class for model predictors.

    Args:
        cfg (DictConfig): Configuration object.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_keys = cfg.MODEL.output_keys
        self.enable_mkldnn = getattr(cfg.INFER, "enable_mkldnn", False)
        self.enable_memory_optim = getattr(cfg.INFER, "enable_memory_optim", True)
        self.enable_ir_optim = getattr(cfg.INFER, "enable_ir_optim", True)
        self.cpu_threads = getattr(cfg.INFER, "cpu_threads", 4)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model from exported files."""
        model_dir = os.path.dirname(self.cfg.INFER.export_path)
        model_prefix = os.path.basename(self.cfg.INFER.export_path)

        logger.message(f"Loading model from {self.cfg.INFER.export_path}")

        # Create config
        config = paddle_infer.Config()

        # Check if model files exist
        if paddle.framework.use_pir_api():
            model_file = os.path.join(model_dir, model_prefix + ".json")
            params_file = os.path.join(model_dir, model_prefix + ".pdiparams")
            if not os.path.exists(model_file) or not os.path.exists(params_file):
                raise ValueError(f"Model files not found: {model_file}, {params_file}")
            # For PIR API
            config.set_prog_file(model_file)  # Just use set_prog_file in both cases
            config.set_params_file(params_file)
        else:
            model_file = os.path.join(model_dir, model_prefix + ".pdmodel")
            params_file = os.path.join(model_dir, model_prefix + ".pdiparams")
            if not os.path.exists(model_file) or not os.path.exists(params_file):
                raise ValueError(f"Model files not found: {model_file}, {params_file}")
            config.set_prog_file(model_file)
            config.set_params_file(params_file)

        # CPU configurations - expand with more optimization options
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(self.cpu_threads)

        # 添加性能优化配置 (安全的API调用方式)
        if self.enable_mkldnn:
            try:
                if hasattr(config, "enable_mkldnn"):
                    config.enable_mkldnn()
                    logger.message("MKL-DNN acceleration enabled")
            except Exception as e:
                logger.warning(f"Failed to enable MKL-DNN: {e}")

        if self.enable_memory_optim:
            try:
                if hasattr(config, "enable_memory_optim"):
                    config.enable_memory_optim()
                    logger.message("Memory optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable memory optimization: {e}")

        if self.enable_ir_optim:
            try:
                if hasattr(config, "switch_ir_optimize") or hasattr(
                    config, "switch_ir_optim"
                ):
                    # 不同版本API名称可能不同
                    if hasattr(config, "switch_ir_optimize"):
                        config.switch_ir_optimize(True)
                    elif hasattr(config, "switch_ir_optim"):
                        config.switch_ir_optim(True)
                    logger.message("IR optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable IR optimization: {e}")

        # Create predictor with enhanced configuration
        self.predictor = paddle_infer.create_predictor(config)
        logger.message("Predictor created successfully with optimized configuration")

    def predict(self, input_dict: Dict, batch_size: int = 64) -> Dict:
        """Predicts the output of the model for a given input.

        Args:
            input_dict (Dict): Input data.
            batch_size (int, optional): Batch size for prediction. Defaults to 64.

        Returns:
            Dict: Predicted output.
        """
        # This is a base method, should be implemented by derived classes
        raise NotImplementedError

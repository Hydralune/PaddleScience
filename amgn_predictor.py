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

from typing import TYPE_CHECKING

from typing import Dict

import numpy as np

from ppsci.deploy.base_predictor import Predictor

from ppsci.utils import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    import pgl


class AMGNPredictor(Predictor):
    """Predictor for AMGNet model.

    Args:
        cfg (DictConfig): Configuration object.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def predict(
        self,
        input_dict: Dict[str, "pgl.Graph"],
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """Predicts the output of the model for a given input.

        Args:
            input_dict (Dict[str, "pgl.Graph"]): Input data in a dictionary.
            batch_size (int, optional): Batch size for prediction. Defaults to 64.

        Returns:
            Dict[str, np.ndarray]: Predicted output in a dictionary.
        """
        # Note: amgnet only supports batch_size=1
        if batch_size > 1:
            logger.warning(
                f"AMGNet predictor only support batch_size=1, but got {batch_size}. "
                "Automatically set batch_size to 1."
            )
            batch_size = 1

        output_dict = {}
        for key, graph in input_dict.items():
            input_names = self.predictor.get_input_names()
            for name in input_names:
                handle = self.predictor.get_input_handle(name)
                data = getattr(graph, name)
                handle.copy_from_cpu(data)

            self.predictor.run()
            output_names = self.predictor.get_output_names()
            for name in output_names:
                handle = self.predictor.get_output_handle(name)
                output = handle.copy_to_cpu()
                output_dict[name] = output

        # mapping data to cfg.INFER.output_keys
        output_dict = {
            store_key: output_dict[infer_key]
            for store_key, infer_key in zip(
                self.output_keys, self.predictor.get_output_names()
            )
        }
        return output_dict

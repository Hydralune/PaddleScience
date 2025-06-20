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

import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.inference as paddle_infer
from matplotlib import pyplot as plt

from ppsci.deploy.base_predictor import Predictor
from ppsci.utils import logger

if TYPE_CHECKING:
    import pgl
    from omegaconf import DictConfig


class AMGNPredictor(Predictor):
    """AMGNet model predictor for inference.
    
    Supports model performance analysis and result visualization.
    """

    def __init__(
        self,
        model_path: str,
        config_params: Optional[Dict] = None,
        verbose: bool = False,
    ):
        """Initialize AMGNet predictor.

        Args:
            model_path: Path to model files.
            config_params: Optional configuration parameters.
            verbose: Whether to print detailed information.
        """
        # Create a minimal config for base class initialization
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "MODEL": {"output_keys": ("pred",)},
            "INFER": {
                "export_path": model_path,
                "enable_mkldnn": config_params.get("use_mkldnn", True) if config_params else True,
                "enable_memory_optim": False,  # Disable to avoid compatibility issues
                "enable_ir_optim": config_params.get("ir_optim", True) if config_params else True,
                "cpu_threads": 4,
            }
        })
        
        logger.message("Initializing AMGNPredictor...")
        super().__init__(cfg)
        self.verbose = verbose
        
        # Initialize input handles for easier access
        self.input_handles = [self.predictor.get_input_handle(name) 
                             for name in self.predictor.get_input_names()]
        
        logger.message("AMGNPredictor initialized successfully.")

    def predict(self, inputs: Dict[str, np.ndarray], batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Run prediction with AMGNet model.

        Args:
            inputs: Input tensor dictionary with keys matching model input names.
            batch_size: Batch size for inference.

        Returns:
            Output tensor dictionary.
        """
        if self.verbose:
            logger.message(f"Node feature shape: {inputs['node_feat'].shape}")

        # Warm up inference engine
        logger.message("Warming up inference engine...")
        self._run_inference(inputs)
        
        # Measure inference time
        start_time = time.time()
        output = self._run_inference(inputs)
        inference_time = time.time() - start_time
        
        node_count = inputs['node_feat'].shape[0] 
        logger.message(f"Inference time: {inference_time:.4f} seconds")
        logger.message(f"Nodes processed: {node_count}")
        logger.message(f"Processing speed: {node_count / inference_time:.2f} nodes/second")
        
        logger.message(f"Prediction successful: {list(output.keys())}")
        return output

    def _run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Internal method to run actual inference.

        Args:
            inputs: Input tensor dictionary.

        Returns:
            Output tensor dictionary.
        """
        results = {}
        
        # Set input data
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            if i == 0:  # Assume first input is node_feat
                handle = self.predictor.get_input_handle(name)
                handle.copy_from_cpu(inputs["node_feat"])
        
        # Run inference
        self.predictor.run()
        
        # Get output
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        
        # Use key 'pred' for consistency with original model
        results["pred"] = output_data
        
        return results

    def input_tensor(self, handle, data):
        """Helper method to copy data to input handle.
        
        Args:
            handle: Paddle inference input handle.
            data: Input data as numpy array.
            
        Returns:
            The input handle after copying data.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        handle.copy_from_cpu(data)
        return handle

    def analyze_results(self, prediction: np.ndarray, ground_truth: np.ndarray):
        """Analyze prediction results compared to ground truth.

        Args:
            prediction: Predicted values.
            ground_truth: Ground truth values.
        """
        # Calculate error metrics
        abs_error = np.abs(prediction - ground_truth)
        rel_error = abs_error / (np.abs(ground_truth) + 1e-10)
        
        # Print summary statistics
        logger.message("\n===== Result Analysis =====")
        logger.message(f"Mean Absolute Error: {np.mean(abs_error):.6f}")
        logger.message(f"Max Absolute Error: {np.max(abs_error):.6f}")
        logger.message(f"Mean Relative Error: {np.mean(rel_error):.6f}")
        logger.message(f"Max Relative Error: {np.max(rel_error):.6f}")
        
        # Calculate per-channel statistics
        for i in range(prediction.shape[1]):
            channel_abs_error = np.abs(prediction[:, i] - ground_truth[:, i])
            channel_rel_error = channel_abs_error / (np.abs(ground_truth[:, i]) + 1e-10)
            logger.message(f"\nChannel {i} statistics:")
            logger.message(f"  Mean Absolute Error: {np.mean(channel_abs_error):.6f}")
            logger.message(f"  Max Absolute Error: {np.max(channel_abs_error):.6f}")
            logger.message(f"  Mean Relative Error: {np.mean(channel_rel_error):.6f}")
            logger.message(f"  Max Relative Error: {np.max(channel_rel_error):.6f}")
            
        # Error histograms
        self._plot_error_histogram(abs_error, "Absolute Error")
        self._plot_error_histogram(rel_error, "Relative Error")
    
    def _plot_error_histogram(self, error_data: np.ndarray, title: str, bins: int = 50):
        """Plot histogram of errors.

        Args:
            error_data: Error data to plot.
            title: Plot title.
            bins: Number of histogram bins.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(error_data.flatten(), bins=bins)
        plt.title(f"{title} Distribution")
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("./result/analysis", exist_ok=True)
        plt.savefig(f"./result/analysis/{title.lower().replace(' ', '_')}_histogram.png")
        plt.close()
        
    def visualize_comparison(
        self, 
        prediction: np.ndarray, 
        ground_truth: np.ndarray, 
        coords: np.ndarray,
        channel_names: List[str] = None
    ):
        """Visualize comparison between prediction and ground truth.

        Args:
            prediction: Predicted values.
            ground_truth: Ground truth values.
            coords: Coordinates for each node.
            channel_names: Names for each channel.
        """
        if channel_names is None:
            channel_names = [f"Channel_{i}" for i in range(prediction.shape[1])]
        
        # Create visualization directory
        os.makedirs("./result/comparison", exist_ok=True)
        
        # Plot each channel
        for i in range(prediction.shape[1]):
            plt.figure(figsize=(15, 7))
            
            # Plot ground truth
            plt.subplot(1, 3, 1)
            sc = plt.scatter(coords[:, 0], coords[:, 1], c=ground_truth[:, i], cmap='viridis', s=5)
            plt.colorbar(sc)
            plt.title(f"Ground Truth - {channel_names[i]}")
            plt.grid(True, alpha=0.3)
            
            # Plot prediction
            plt.subplot(1, 3, 2)
            sc = plt.scatter(coords[:, 0], coords[:, 1], c=prediction[:, i], cmap='viridis', s=5)
            plt.colorbar(sc)
            plt.title(f"Prediction - {channel_names[i]}")
            plt.grid(True, alpha=0.3)
            
            # Plot absolute error
            plt.subplot(1, 3, 3)
            error = np.abs(prediction[:, i] - ground_truth[:, i])
            sc = plt.scatter(coords[:, 0], coords[:, 1], c=error, cmap='hot', s=5)
            plt.colorbar(sc)
            plt.title(f"Absolute Error - {channel_names[i]}")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"./result/comparison/{channel_names[i]}_comparison.png")
            plt.close()
        
        logger.message(f"Visualizations saved to ./result/comparison") 
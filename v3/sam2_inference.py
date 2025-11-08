import torch
import numpy as np

# Core framework imports
from core import ModelConfig, DeviceMesh

# from core.model_builder import CustomModel

# SAM2 utilities (for video predictor wrapper)
# from models.sam2.layers.video_predictor import VideoPredictor  # placeholder

model_cfg_path = "models/sam2/config.yaml"
checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"

config = ModelConfig(model_cfg_path)

device_mesh = DeviceMesh(["cuda:0", "cuda:1"])

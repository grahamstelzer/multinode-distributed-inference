import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


# setup visualization functions
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()



# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )





import yaml, json
from pathlib import Path

cfg_path = Path(__file__).resolve().parent / "sam2.1_hiera_l.yaml"
json_path = cfg_path.with_suffix(".json")

with open(cfg_path) as f:
    cfg_yaml = yaml.safe_load(f)

with open(json_path, "w") as f:
    json.dump(cfg_yaml, f, indent=2)

print(f"Config converted → {json_path}")

import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam2_utils import MLP, LayerNorm2d
from sam2.modeling.position_encoding import PositionEmbeddingRandom
from sam2.utils.transforms import SAM2Transforms

# image setup
image = Image.open('./truck.jpg').convert("RGB")
image = np.array(image)

# weight/config path setup
repo_root = Path(__file__).resolve().parent
sam2_checkpoint= repo_root / "sam2.1_hiera_large.pt"
cfg_path = repo_root / "sam2.1_hiera_l.json"

# check
assert sam2_checkpoint.exists(), "Checkpoint missing!"
assert cfg_path.exists(), "Config missing!"

with open(cfg_path) as f:
    cfg = json.load(f) # NOTE: cfg is nested python dict

print(cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load checkpoint weights, dependant on existence of 'model_state_dict' key:
checkpoint = torch.load(sam2_checkpoint, map_location=device)
weights = checkpoint.get("model_state_dict", checkpoint)




backbone = build_backbone(cfg['model']['backbone'])
prompt_encoder = PromptEncoder(**cfg['model']['prompt_encoder'])
mask_decoder = MaskDecoder(**cfg['model']['mask_decoder'])

sam2_model = SAM2Base(
    backbone=backbone,
    prompt_encoder=prompt_encoder,
    mask_decoder=mask_decoder,
    pixel_mean=cfg['model']['pixel_mean'],
    pixel_std=cfg['model']['pixel_std'],
)

sam2_model.load_state_dict(weights, strict=False)
sam2_model.to(device)






from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print("Masks:", masks.shape)
print("Scores:", scores)


# for name, module in sam2_model.named_modules():
#     print(name, type(module))

# if isinstance(module, nn.MultiheadAttention):
#     setattr(parent, name, DistributedAttention(...))
import torch
import numpy as np
from PIL import Image
from sam2_image_predictor import SAM2ImagePredictor

# Path to your pretrained SAM2 weights (check sam2 README for the filename)
MODEL_ID = "./checkpoints/sam2.1_hiera_large.pt"  # or path to local checkpoint

def main():
    # Load predictor from pretrained
    predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID)

    # Load a test image
    img = Image.open("cars.jpg").convert("RGB")

    # Register hooks/prints are already in your code.
    predictor.set_image(img)

    # Run a dummy prediction – here just a single foreground point in the middle
    h, w = img.size
    point_coords = np.array([[w // 2, h // 2]])   # center point
    point_labels = np.array([1])                  # 1 = foreground

    masks, ious, low_res = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )

    print("masks:", masks.shape, "ious:", ious, "low_res:", low_res.shape)

if __name__ == "__main__":
    main()

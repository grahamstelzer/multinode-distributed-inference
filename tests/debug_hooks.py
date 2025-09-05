import torch
import inspect
import numpy as np
from PIL import Image

# ---- Import SAM2 predictor ----
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---- Hook utilities ----
def hook_fn(module, input, output):
    """Print info for Linear, Attention, Conv2d, MLP modules."""
    stack = inspect.stack()
    caller = stack[2].function  # get function that invoked forward
    print(f"\n[HOOK] {module.__class__.__name__} called from `{caller}`")

    if isinstance(output, torch.Tensor):
        print(" -> output:", tuple(output.shape), "device:", output.device)
    elif isinstance(output, (list, tuple)):
        shapes = [tuple(o.shape) for o in output if isinstance(o, torch.Tensor)]
        print(" -> outputs:", shapes)

# ---- Main ----
def main():
    # Load pretrained SAM2 (replace with your model_id if different)
    model_id = "facebook/sam2-hiera-large"
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    model = predictor.model

    # Attach hooks
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.MultiheadAttention,
                               torch.nn.Linear,
                               torch.nn.Conv2d)):
            module.register_forward_hook(hook_fn)

    print("\n✅ Hooks registered. Running dummy inference...")

    # Make dummy image
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy)

    # Run inference
    predictor.set_image(dummy_img)
    masks, scores, lowres = predictor.predict(
        point_coords=np.array([[128, 128]]),
        point_labels=np.array([1])
    )

    print("\n🎉 Done. Shapes of outputs:")
    print("Masks:", masks.shape)
    print("Scores:", scores.shape)
    print("Lowres masks:", lowres.shape)

if __name__ == "__main__":
    main()

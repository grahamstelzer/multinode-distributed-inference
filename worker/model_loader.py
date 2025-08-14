# model_loader.py
# Responsible for loading models and weights (pseudo for now)

class ModelLoader:
    def __init__(self):
        # For PoC, we just simulate a loaded model
        self.model = "pseudo_model_loaded"

    def load_model(self, model_name=None):
        """
        Load a model into GPU memory.
        For now, just return a string representing the model.
        """
        print(f"[MODEL_LOADER] Loading model: {model_name}")
        return self.model

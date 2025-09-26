from transformers import AutoConfig, AutoModel

MODEL = "facebook/sam2-hiera-large"

config = AutoConfig.from_pretrained(MODEL)
print(config)   # dumps hyperparameters

model = AutoModel.from_pretrained(MODEL)
print(model)    # prints torch.nn.Module architecture

from transformers import AutoConfig, AutoModel

MODEL = "facebook/sam2-hiera-large"


config = AutoConfig.from_pretrained(MODEL)
with open("config.txt", "w") as f:
    f.write(str(config))

model = AutoModel.from_pretrained(MODEL)
with open("model.txt", "w") as f:
    f.write(str(model))    # prints torch.nn.Module architecture



from torchinfo import summary
summary(model, input_size=(1, 3, 1024, 1024))
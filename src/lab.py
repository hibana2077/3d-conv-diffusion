from timm.models.inception_v4 import InceptionV4

model = InceptionV4()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
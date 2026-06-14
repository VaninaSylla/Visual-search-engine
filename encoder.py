import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Chargement ResNet50 pré-entraîné
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # On enlève la dernière couche fully-connected (classification)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)          # [B, 2048, 1, 1]
            features = features.squeeze(-1).squeeze(-1)  # [B, 2048]
        return features

def encode_image(encoder, image_tensor):
    """Encode un tenseur image en vecteur."""
    return encoder(image_tensor).numpy()
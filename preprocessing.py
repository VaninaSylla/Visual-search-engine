import os
from PIL import Image
from torchvision import transforms

# Transformations standard pour un modèle pré-entraîné ImageNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # moyenne ImageNet
        std=[0.229, 0.224, 0.225]     # écart-type ImageNet
    )
])

def load_image(image_path):
    """Charge et prétraite une image depuis son chemin."""
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0)  # ajout dim batch → [1, 3, 224, 224]

def get_all_image_paths(dataset_dir):
    """Retourne tous les chemins d'images .jpg / .png du dataset."""
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from huggingface_hub import login

augmentation_pipeline = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True), # Note: size=384 for fine-tuning
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet-1k mean and std
])

login()
image_stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)

# TODO: find another dataset to fine tune the ViT on
# TODO: apply the augmentation pipeline to the dataset
# TODO: Create a DataLoader based off of the HF dataset
# TODO: Visualize an image using matplotlib
# TODO: Implement the ViT model
# TODO: pretrain model on ImageNet-1k
# TODO: Visualize the attention maps
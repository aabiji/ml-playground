# Read
- An image is worth 16x16 words: Transformers for image recognition at scale
- ViT-5: Vision Transformers for The Mid-2020s
- Rotary Position Embedding for Vision Transformer
- Training data-efficient image transformers & distillation through attention
- Emerging Properties in Self-Supervised Vision Transformers
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

# Implementation
- Original ViT model and the ViT-5 model - compare inference performance between the architectures.
  What's the magnitude of the improvement the architectural changes provide?

- Use model checkpoints to train over several runs.

- Plot text progress bar instead of plotting loss each step, only plot loss at the end

- Data augmentation techniques: random flipping, random rotation, random crop out, random gaussian noise, random noise, random zoom
  - Note that augmentation adds more images to the dataset
  - Visualize the effects of data augmentation

- Inference demos:
  - Show general accuracy, show image, display top 5 model predictions, display the actual prediction, press space to choose another random image
  - Show general accuracy, show image, overlay predicted segmentation mask, overlay the actual segmentation mask, press space to change to another random image

# Experiments
- Visualize positional embeddings. Visualize the attention scores. Visualize the embedding filters.
  What are some interesting patterns?

- Why GeLU? Does it offer substancial performance improvements over ReLU?

- Can we fine tune a ViT that was originally trained on a classification task on a segmentation task? Kind of like how GPT-2 was pretrained
  using next token prediction then different heads were used? Or will the lack of learned localization hinder progress?

- Which loss function is most effective for segmentation? Pixel based cross entropy loss, DICE loss, or Jaccard loss?

- Visualize the model's prediction on a medical dataset to better see how training loss correlates with qualitative performance.

# Next steps
- Mamba for images??

- Learn about DINO to see if I can replicate their core result.

- Look into VAEs

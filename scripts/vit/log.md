# Implementation
- Use model checkpoints to train over several runs. Consider loading a pretrained ViT in order to focus on fine-tuning.

- Inference demos:
  - Show general accuracy, show image, display top 5 model predictions, display the actual prediction, press space to choose another random image
  - Show general accuracy, show image, overlay predicted segmentation mask, overlay the actual segmentation mask, press space to change to another random image

# Ideas
- Visualize and see if there any patterns:
  - Positional embeddings
  - Embedding filters

- Pretrain a Vit on CIFAR10:
  - Fine tune on ImageNet. That would increase the image resolution from 32x32 to 224x224,
    and that would also change the number of labels, from 10 to 1000.
    Will this attempt at transfer learning result in improved classification performance.

  - Replace the classification head with a segmentation head. Then, fine-tune on COCO minitrain.
    This is inspired by the way that different heads could be swapped out for different tasks in GPT-2.
    Will the lack of localized semantic understand hinger performance, or can localization be learned
    quickly if there's already a semantic understanding?
    - Experiment with pixel based cross-entropy loss, DICE loss or Jaccard loss to see which is most effective

- Implement the improvements proposed in ViT-5 and compare performance with the base ViT model

- Read the knowledge distillation paper and see if a large pretrained ViT model
  (download the weights from somewhere else), can be distilled into a 100 M (at most) param model.

# Notes
**(Touvron et al., 2020)**
- (Touvron et al., 2019) show that it's desirable to use a lower image resolution during training and a higher image
  resolution during fine tuning. Positional embeddings are thus interpolated to accomodate the larger token sequence.
- Interesting that they didn't find Dropout to be useful. They used 300 epochs, 1024 batch size, AdamW optimizer,
  0.0001 learning rate, cosine learning rate decay, 0.05 weight decay, 5 warmup epochs, 0.1 label smoothing.

**(Wang et al., 2025)**

# Progress
September 26, 2026:
- Read *An Image is Worth 16x16 words: Transformers for image recognition at scale*
- Implemented a basic version of the original ViT: MSA, transformer layers
- Loaded and visualized the CIFAR10 dataset, and converted images into patches
- Need to read *ViT-5: Vision Transformers for The Mid-2020s* to get better of what training recipe I should use

September 28, 2026:
- Read *Training data-efficient image transformers & distillation through attention*
- Small improvements on the model

September 30, 2026:
- Finished implementing the core trianing loop
- Visualized image patches and attention scores
- **How can we visualize the Q, K, V, O projections and the positional embeddings?**
  given that they are very high dimensional

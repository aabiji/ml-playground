# Implementation
- Inference demos:
  - Show general accuracy, show image, display top 5 model predictions, display the actual prediction, press space to choose another random image
  - Show general accuracy, show image, overlay predicted segmentation mask, overlay the actual segmentation mask, press space to change to another random image

# Ideas
- Pretrain a Vit on CIFAR10:
  - Fine tune on ImageNet. That would increase the image resolution from 32x32 to 224x224,
    and that would also change the number of labels, from 10 to 1000.
    Will this attempt at transfer learning result in improved classification performance.

  - Replace the classification head with a segmentation head. Then, fine-tune on COCO minitrain.
    This is inspired by the way that different heads could be swapped out for different tasks in GPT-2.
    Will the lack of localized semantic understand hinger performance, or can localization be learned
    quickly if there's already a semantic understanding?

    - Read to better understand how segmentation with ViTs are done:
      - https://ai.stackexchange.com/questions/46002/vision-transformer-for-image-segmentation
      - https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformers-for-image-segmentation
      - https://openaccess.thecvf.com/content/CVPR2025/papers/Kerssies_Your_ViT_is_Secretly_an_Image_Segmentation_Model_CVPR_2025_paper.pdf

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
- The ViT architecture remains under-optimized but improvements are not orthogonal.
- Improvements:
  - Although LayerNorm keeps the residual output range roughly constant, the outputs themselves get larger.
    LayerScale uses a learned weight to scale down residual outputs so that they don't explode deeper into the network.
    $x_{i + 1} = x_i + F(x) \odot \lambda$, where $\lambda \in \real^d$ initialized to something like $10^-4$,
    and $F(x)$ is a MSA or MLP output.

# Progress
September 26, 2026:$
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
- How can we visualize the Q, K, V, O projections and the positional embeddings? given that they are very high dimensional

October 1, 2026:
- First experiment ran successfully and I took some notes on how to improve my implementation. Mostly implementation errors.
- Implemented the various improvement ideas: improve plotting, fix [class] embedding prepending bug, visualizing positional
  embeddings, adding an actual classification head instead of just a projection, etc.
- Measured top-1 accuracy, and switch to the STL10 dataset to test out larger image sizes.
- Tweaks hyperparams to build a bigger model.

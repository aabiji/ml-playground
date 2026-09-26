*Learn real analysis as another step after calc??*

[Neural Net Training Recipe](https://karpathy.github.io/2019/04/25/recipe/)

ViT exploration:
- Classification task
  - Compare validation losses for models with increasing patch sizes to determine how much patch size matters.
    Do these general observations hold for different image sizes?
    If increasing the patch size is beneficial, then I expect it would be capped by memory and compute constraints.

  - How much does positional encoding matter? ViTs perform worse without any positional encoding because they have
    no inductive bias for neighboring pixels, unlike CNNs. But how much does perf improve if we switch from absolute
    1D positional encoding to 2D absolute encoding to 1D relative rotary positional encoding?

  - Can we visualize the attention scores from all heads at different transformer layers? Are there any patterns?
    Are they similar in nature to the convolutional filters of CNNs? How much do they differ at later layers?

  - How much does data augmentation actually help?
    Compare validation loss with and without iamge flipping, random erasure, blur and noise.

- Segmentation task
  - Which loss function is most effective? Pixel based cross entropy loss, DICE loss, or Jaccard loss?

  - Visualize the model's prediction on a medical dataset to better see how training loss correlates with qualitative performance.

  - Learn about DINO to see if I can replicate their core result.

- What are some of the SOTA ViT architectures and techniques?

# Next steps
- Replace the classification head with a segmentation head. Then, fine-tune on COCO minitrain.
  This is inspired by the way that different heads could be swapped out for different tasks in GPT-2.
  Will the lack of localized semantic understand hinger performance, or can localization be learned
  quickly if there's already a semantic understanding?

  - Read to better understand how segmentation with ViTs are done:
    - https://ai.stackexchange.com/questions/46002/vision-transformer-for-image-segmentation
    - https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformers-for-image-segmentation
    - https://openaccess.thecvf.com/content/CVPR2025/papers/Kerssies_Your_ViT_is_Secretly_an_Image_Segmentation_Model_CVPR_2025_paper.pdf

  - Experiment with pixel based cross-entropy loss, DICE loss or Jaccard loss to see which is most effective

- Read, understand and implement the VAE paper

- Join a Kaggle competition if I can't find a prof to work with by the end of the month.

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
- How can we visualize the Q, K, V, O projections and the positional embeddings? given that they are very high dimensional

October 1, 2026:
- First experiment ran successfully and I took some notes on how to improve my implementation. Mostly implementation errors.
- Implemented the various improvement ideas: improve plotting, fix [class] embedding prepending bug, visualizing positional
  embeddings, adding an actual classification head instead of just a projection, etc.
- Measured top-1 accuracy, and switch to the STL10 dataset to test out larger image sizes.
- Tweaks hyperparams to build a bigger model.

October 2, 2026:
- The visualized attention scores don't actually tell me much, so I'm axing that. The positional similarities are interesting though,
  patches appear in the heatmap corresponding to semantically similar regions in the image.
- Trained a larger model (6.2 M params -> 101 M params), accuracy is still very poor, only 30.20% top-1 accuracy on the STL10 dataset.
  It's unclear whether the model fails to generalize because of suboptimal hyperparameters, model architecture or training dynamics...will need to read some more literature

October 4, 2026:
- Did a first pass of (Wang et al., 2025) and (Shazeer, 2020). Still a lot more depth to uncover.
- Came up with a few next steps:
  - Visualize how attention scores changing throughout model training. How are the attention scores in the papers even computed?
  - Compare and contrast test performance with LayerScale, 2D ROPE and RMSNorm added.
  - Can we distill a large ViT-L model into a much tinier model? How small is too small?
  - Train a different combinations of hyperparams to see if they are the issue.
  - Attempt to overfit on a tiny dataset with a tiny model. How would I even know if I'm successful?
  - What if patch tokens were 1x1? How does perf change?
  - Should be plotting validation loss, not training loss.

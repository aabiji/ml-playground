import loader
import model
import matplotlib.pyplot as plt
import torch

#dataset = loader.LibriSpeech(".cache/LibriSpeech/dev-clean")
#spectrograms, _, transcripts, _ = dataset[0:2]
#loader.spec_augment(spectrograms, 27, 1, 100, 1, 0.0)
#plt.figure(figsize=(10, 6))
#plt.imshow(spectrograms[0], aspect="auto", origin="lower")
#plt.savefig("out.png", bbox_inches="tight", pad_inches=0.1)

#layer_ff_dims = [512, 768, 768, 768, 768, 768]
#embed_dims = [192, 256, 256, 256, 256, 256]
#layer_counts = [2, 2, 2, 2, 2, 2]
#attn_heads = [4, 4, 4, 8, 4, 4]
#conv_kernel_sizes = [31, 31, 15, 15, 15, 31]
#sampling_factors = [2, 4, 8, 4, 2, 2]
#Zipformer(80, conv_kernel_sizes, attn_heads, layer_ff_dims, embed_dims, layer_counts, sampling_factors)

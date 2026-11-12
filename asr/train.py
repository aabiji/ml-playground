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

x = torch.tensor([[[1, 2, 3], [3, 4, 5]]])
bias = torch.arange(3)
model.biasnorm(x, bias, 0.1)
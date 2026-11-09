import loader
import matplotlib.pyplot as plt

dataset = loader.LibriSpeech(".cache/LibriSpeech/dev-clean")
spectrograms, _, transcripts, _ = dataset[0:2]
plt.imshow(spectrograms[0])
plt.savefig("out.png")
print(labels)

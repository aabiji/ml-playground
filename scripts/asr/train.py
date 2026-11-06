# %%
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class LibriSpeech(Dataset):
  def __init__(self, folder_path):
    self.mel_transform = MelSpectrogram(
      sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )  # 10 ms window stride
    self.db_transform = AmplitudeToDB()

    files = [file for file in Path(folder_path).rglob("*") if file.is_file()]
    self.audio_files = list(filter(lambda f: f.suffix == ".flac", files))
    self.transcript_cache = {}

    # Cache transcripts
    transcript_files = list(filter(lambda f: f.suffix == ".txt", files))
    for tfile in transcript_files:
      key = tfile.stem.split(".")[0]
      with open(tfile, "r") as f:
        lines = f.read().split("\n")
        self.transcript_cache[key] = lines

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    elif isinstance(idx, slice):
      idx = torch.arange(len(self.audio_files))[idx]
    else:
      idx = [idx]

    waveforms = []
    transcripts = []
    for i in idx:
      f = self.audio_files[i]
      waveform, _ = torchaudio.load(f, normalize=True)
      waveform = waveform[0].permute(1, 0)  # (channels, time) -> (time, channels)
      waveforms.append(waveform)

      last_dash = f.stem.rfind("-")
      key, index = f.stem[:last_dash], int(f.stem[last_dash + 1 :])
      words = self.transcript_cache[key][index].split(" ")
      transcript = " ".join(words[1:])
      tokens = list(map(ord, transcript))  # Codepoint-level tokenization
      transcripts.append(torch.tensor(tokens))

    # (batch size, time, channels) -> (batch size, channels, time)
    sample_batch = pad_sequence(waveforms, batch_first=True, padding_value=0)
    sample_batch = sample_batch.permute(1, 2, 0)

    sample_batch = self.mel_transform(sample_batch)
    sample_batch = self.db_transform(sample_batch)
    widths = list(map(waveforms, key=lambda w: w.shape[-1]))

    label_batch = pad_sequence(transcripts, batch_first=True, padding_value=0)
    return sample_batch, label_batch, widths


def spec_augment(spec_batch, max_width, max_height, mask_value):
  b, h, w = spec_batch.shape
  idx = torch.arange(b)
  rand_w = torch.randint(0, max_width, size=(b,))
  rand_h = torch.randint(0, max_height, size=(b,))
  rand_x = torch.tensor(
    [torch.randint(0, w - rand_w[i].item(), size=(1,)) for i in range(b)]
  )
  rand_y = torch.tensor(
    [torch.randint(0, h - rand_h[i].item(), size=(1,)) for i in range(b)]
  )
  spec_batch[idx, rand_x : rand_x + rand_w, rand_y : rand_y + rand_h] = mask_value


dataset = LibriSpeech(".cache/LibriSpeech/dev-clean")
sample_batch, label_batch = dataset[0:2]
print(spectrogram.shape)
plt.imshow(spectrogram[0])
plt.savefig("out.png")

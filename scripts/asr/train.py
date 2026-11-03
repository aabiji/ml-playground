# %%
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchaudio
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
    assert not torch.is_tensor(idx)
    f = self.audio_files[idx]

    last_dash = f.stem.rfind("-")
    key, index = f.stem[:last_dash], int(f.stem[last_dash + 1 :])
    words = self.transcript_cache[key][index].split(" ")
    transcript = " ".join(words[1:])

    waveform, _ = torchaudio.load(f, normalize=True)
    spectrogram = self.mel_transform(waveform)
    spectrogram = self.db_transform(spectrogram)
    return spectrogram, transcript


dataset = LibriSpeech(".cache/LibriSpeech/dev-clean")
spectrogram, transcript = dataset[0]
plt.imshow(spectrogram[0])

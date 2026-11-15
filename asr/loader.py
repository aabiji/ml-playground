import librosa
import numpy as np
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def spec_augment(spec_batch, max_freq_range, n_freq_masks,
                 max_time_range, n_time_masks, mask_value):
  _, spec_height, spec_width = spec_batch.shape

  for _ in range(n_freq_masks):
    f_range = torch.randint(0, max_freq_range, size=(1,))
    y = torch.randint(0, spec_height - f_range, size=(1,))
    spec_batch[:, y:y+f_range, :] = mask_value

  for _ in range(n_time_masks):
    t_range = torch.randint(0, max_time_range, size=(1,))
    x = torch.randint(0, spec_width - t_range, size=(1,))
    spec_batch[:, :, x:x+t_range] = mask_value

class LibriSpeech(Dataset):
  def __init__(self, folder_path):
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

  def process_all_samples(self, all_samples):
    # Convert a batch of audio samples into a batch of padded spectrograms with a
    # framerate of 100 kHz. For example, with 1000 ms of audio, a 16 kHz sampling
    # rate and a hop length of 160 samples, that's (1000 * 16000) / 160 + 1 = 101
    # time frames. Adding 1 since each frame is centered on its timestamp rather
    # than starting on it.
    hop_length = 160
    samples_lengths = list(map(lambda x: x.shape[-1], all_samples))

    frame_lengths = (torch.tensor(samples_lengths) // hop_length) + 1

    # Padding samples before applying the spectrogram for efficiency
    padded = pad_sequence(all_samples, batch_first=True, padding_value=0)
    padded_np = padded.detach().cpu().numpy()

    spectrograms = librosa.feature.melspectrogram(y=padded_np, sr=16000, hop_length=hop_length,
                                                  n_mels=80, n_fft=400, win_length=400)
    spectrograms = librosa.power_to_db(spectrograms, amin=1e-5, ref=np.max)
    spectrograms = torch.from_numpy(spectrograms).float()
    return spectrograms, frame_lengths

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, raw_idx):
    all_samples = []
    transcripts = []
    tok_counts = []

    if torch.is_tensor(raw_idx):
      idx = raw_idx.tolist()
    elif isinstance(raw_idx, slice):
      idx = torch.arange(len(self.audio_files))[raw_idx].tolist()
    else:
      idx = [raw_idx]

    for i in idx:
      # Load audio sample
      f = self.audio_files[i]
      np_samples, _ = librosa.load(f, mono=True, sr=16000)
      all_samples.append(torch.from_numpy(np_samples).float())

      # Get the corresponding transcript and apply codepoint-level tokenization
      last_dash = f.stem.rfind("-")
      key = f.stem[:last_dash]
      line_index =  int(f.stem[last_dash + 1:])
      words = self.transcript_cache[key][line_index].split(" ")

      tokens = list(map(ord, " ".join(words[1:])))
      transcripts.append(torch.tensor(tokens))
      tok_counts.append(len(tokens))

    transcripts = pad_sequence(transcripts, batch_first=True, padding_value=0)
    transcript_lengths = torch.tensor(tok_counts)
    spectrograms, frame_lengths = self.process_all_samples(all_samples)
    return spectrograms, frame_lengths, transcripts, transcript_lengths

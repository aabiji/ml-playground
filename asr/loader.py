import librosa
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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
  spec_batch[idx, rand_x:rand_x+rand_w, rand_y:rand_y+rand_h] = mask_value

class LibriSpeech(Dataset):
  def __init__(self, folder_path):
    # 16 kHz sampling rate, 10 ms window stride, -100 dB floor
    self.sample_rate, self.hop_length, self.db_floor = 16000, 160, 1e-5
    #self.melspec = MelSpectrogram(
    #
    #)
    #self.to_db = AmplitudeToDB()
    #self.to_db.amin = self.db_floor

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

  def process_waveforms(self, waveforms, sample_lengths):
    frame_lengths = torch.tensor(sample_lengths) // self.hop_length + 1

    padded = pad_sequence(waveforms, batch_first=True, padding_value=0)
    padded_np = padded.detach().cpu().numpy()

    spectrograms = librosa.feature.melspectrogram(y=padded_np, sr=16000, hop_length=160,
                                                  n_mels=80, n_fft=400, win_length=400)
    spectrograms = librosa.power_to_db(spectrograms, ref=torch.max)
    spectrograms = torch.from_numpy(spectrograms).float()

    print(spectrograms.shape)
    return spectrograms, frame_lengths

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, raw_idx):
    waveforms, sample_lengths = [], []
    transcripts,  tok_counts = [], []

    if torch.is_tensor(raw_idx):
      idx = raw_idx.tolist()
    elif isinstance(raw_idx, slice):
      idx = torch.arange(len(self.audio_files))[raw_idx].tolist()
    else:
      idx = [raw_idx]

    for i in idx:
      # Load audio sample
      f = self.audio_files[i]
      samples, sample_rate = librosa.load(f, mono=True, sr=16000)
      sample_lengths.append(samples.shape[-1])
      waveforms.append(torch.from_numpy(samples).float())

      # Get the corresponding transcript and apply codepoint-level tokenization
      last_dash = f.stem.rfind("-")
      key = f.stem[:last_dash]
      line_index =  int(f.stem[last_dash + 1:])
      words = self.transcript_cache[key][line_index].split(" ")

      transcript = " ".join(words[1:])
      tokens = list(map(ord, transcript))
      transcripts.append(torch.tensor(tokens))
      tok_counts.append(len(tokens))

    transcripts = pad_sequence(transcripts, batch_first=True, padding_value=0)
    transcript_lengths = torch.tensor(tok_counts)
    spectrograms, frame_lengths = self.process_waveforms(waveforms, sample_lengths)
    return spectrograms, frame_lengths, transcripts, transcript_lengths

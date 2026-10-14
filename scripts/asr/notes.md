Monolingual ASR Exploration Project
Target completion date: November 15, 2026

**Done**
Read [Measuring benchmark optimization in speech recognition](https://huggingface.co/blog/asr-benchmark-optimization).
Some models on ASR benchmarks have been knowingly or unknowingly trained on benchmark datasets. This results in them
outputting verbatim what the benchmark states is the correct answer for a samples, instead of what was actually spoken.
Addtionally, a lower WER score doesn't necessarily mean better performance because WER doesn't account for semantic meaning.
For example, a model output that uses the correct words but uses the wrong case will be scored incorrectly.

**Vague TODO list**

- Read [Speech representations: From Waveforms to MEL Spectrograms](https://mbrenndoerfer.com/writing/speech-representations-mel-spectrograms)
- Visualize audio by plotting a MEL spectrogram of a 5s mono 44.1 kHz audio clip. Extend that to dual channels.

- Skim some papers:
  - [OpenWER: Improving Cross-Lingual ASR Evaluation and Enabling Token-Based Accuracy Metrics](https://arxiv.org/html/2606.21237v1)
  - [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/pdf/2608.30320)
  - [Kimi Audio](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)

- Implement the techniques detailed in SpecAugment
  - Read [SpecAugment](https://arxiv.org/pdf/1904.08779)

- Implement a basic CLI real time transcription demo using a pretrained model: Parakeet-TDT-0.6B-v3
  - Skim [Canary-1B-v2 & Parakeet-TDT-0.6B-v3](https://arxiv.org/html/2509.14128v2)
  - Load audio from audio files, and mp3 or wav files.
  - Stream audio in overlapping N second chunks and output the audio using several threads or
    some sort of asynchronous processing
  - Print interim and finalized transcript segments with timestamps
  - Test with continuous speech, speech pauses, background noises and silence

- Attempt to distill the parakeet model (or another similar model) into a smaller model
  - Read [Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531)
  - Inspect the teacher's architecture, tokenizer, output format, training framework and available training checkpoints
  - Create a representative training subset of the audio samples and transcriptions
  - Find a student architecture and a training recipe that I can train locally
    - Experiment: What's the smallest distilled model that will maintain a +/- 10% accuracy difference?
  - ... come up with the next set of tasks as I go ...

- Quantize the distilled model to further improve performance
  - Read [Practical Quantization in Pytorch](https://pytorch.org/blog/quantization-in-practice/)
  - Target CPU/iGPU inference and GPU inference separately
  - Benchmark model size, performance and accuracy at various quantization modes

- Reimplement the Whisper model and train it on a tiny dataset
  - Read [wav2vec 2.0](https://arxiv.org/pdf/2006.11477)
  - Read [Whisper](https://cdn.openai.com/papers/whisper.pdf)
  - Learn how to use Optuna to run a hyperparamter sweep
  - Train a tiny model on a tiny speech dataset

- *Final challenge*: Come up with a custom model architecture and benchmark against
  existing SOTA models. What if we treated audio samples as images, split the mel
  spectrograph in patches, project the patches and apply many of the improvements
  made in recent ViT models one at a time (LayerScale, RMSNorm, ROPE, SwiGLU, etc)?
  Is this even a good idea?  How much of a unique modality is audio?

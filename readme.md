Ideas:
- Train a gpt on shakespear or tiny stories and actually get inference perf
- Recreate grokking modular addition
- Implement tiny ASR model
- Understand and implement DDPM

TinyStories: their dataset has a token space that reflects a smaller, more focused domain, making it easier for the model to do next-token prediction.

ASR Project:
Questions:
- How should I be normalizing spectrograms?

- First phase: implementation
    - Skim the conformer, fast conformer, zipconformer, wav2vec, parakeet, whisper model papers to get some architecture ideas
    - Train our own conformer on librispeech with a ctc head and no language model.
        - Learn about beam search and how its used for ctc inference.
        - Overfit tiny dataset to fix bugs, train on small sample to test out different model ablations, then train final architecture and learning recipe on massive dataset.
    - Attempt to distill parakeet models into a tiner model and compare performance across the original, the the distilled and my own original model.
    - Attempt to fine tune my models on my voice. How sample even are these models?

- Second phase: product
    - Build a voice dictation daemon for my laptop and preferably it would skip words like “period” and “comma", with actual periods and commas (given the appropriate context). And be small enough overhead that I can continually run this or launch the demon repeatedly with a keyboard shortcut.

- Third phase: questions
    - How do the models handle varying noise levels? How can that be improved?
    - Are these models sample efficient? How can they be made more sample efficient?
    - Can these models also determine who is speaking (speaker diarization)?
    - Can we get fast and coherent ctc without an external language model?
    - How about text to speech?
"""
### Improving language understanding by generative pre-training

**Model input**: NxE matrix of token embeddings
- N = context window size, E = embedding dimension
- There's no start token but the context window may include padding tokens and an end of sequence token.
- Context windows are slid over the token corpus.
---
**Model**: Transformer decoder (the masked self attention adds auto-regressivity).
- During inference, the model receives an ever growing chunk of tokens and outputs a single token in a loop. This is inherently sequential 😬.
- Training is split into two phases:
  - **Self supervised pre-training**: The model receives a chunk of tokens and outputs a probability over a chunk of tokens. We want to maximize the probability of outputting the next token in the sequence (the shifted context window).

  - **Supervised fine-tuning**: The model receives a chunk of tokens and outputs a probability over an output label *y*. We want to maximize the probability of outputting the correct label and the next correct output token. The model parameters are transferred and the last linear layer is swapped out.
---
**Experiment ideas**:
- Remove the fine tuning stage and go diretly to text generation
- Overfit on a tiny dataset to make sure that the model is implemented correctly
- Compare validation losses:
  - Different context window sizes
  - Different number of layers
  - Different number of attention heads
  - Remove/replace LayerNorm
  - Remove causal mask
  - Different dataset sizes
  - Different parameter counts
  - Vary temperature and experiment with top-k, top-p, greedy decoding, etc
  - What if we remove the MLP?
  - What if we remove attention?
---
Datasets:
- [wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- [OpenPhi textbooks](https://huggingface.co/datasets/open-phi/textbooks)
- [Youtube comment sentiment](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment)
"""

import torch
import torch.nn.functional as F
import datasets
import tiktoken

def load_dataset(cache_folder):
  try:
    return datasets.load_from_disk(cache_folder)
  except:
    base = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
      "got2",
      pat_str=base._pat_str,
      mergeable_ranks=base._mergeable_ranks,
      special_tokens={"<EOS>": base.max_token_value},
    )
    dataset = datasets.load_dataset("open-phi/textbooks", split="train")
    dataset = dataset.map(
      lambda rows: {"token_ids": [enc.encode(f"{r}<EOS") for r in rows["markdown"]]},
      batched=True,
    )
    dataset.save_to_disk(cache_folder)
    return dataset

torch.manual_seed(67)
ctx_window_size = 1024

dataset = load_dataset(".cache/open-phi-textbooks")

# TODO: now pad everything at once....put all tokens into one big matrix and pad columns to the nearest multiple of ctx_window_size, then split into chunks
tokens = torch.tensor(dataset[0]["token_ids"], dtype=torch.float32)
chunks = list(torch.split(tokens, ctx_window_size))
chunks[-1] = F.pad(chunks[-1], (0, ctx_window_size - chunks[-1].numel()), value=0)
x = torch.stack(chunks)


print(x.shape)
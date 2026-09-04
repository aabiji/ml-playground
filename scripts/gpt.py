from datasets import load_dataset
import tiktoken
import torch
import torch.nn as nn

def get_line_stats(batch, tokenizer):
  tokens_batch = []
  length_batch = []
  for row in batch["text"]:
    tokens = tokenizer.encode(row)
    tokens_batch.append(tokens)
    length_batch.append(len(tokens))
  return {"tokens": tokens_batch, "length": length_batch}

def prepare_dataset(name, cache_file, tokenizer, pad_token, tok_chk_size):
  # Load the dataset and gather token lengths
  dataset = load_dataset(name, split="train")
  dataset = dataset.map(
    lambda b: get_line_stats(b, tokenizer),
    batched=True,
    remove_columns=["text"]
  )

  # Pad each token sequence into a fixed size
  shape = (len(dataset), max(dataset["length"]))
  padded = torch.full(shape, pad_token, dtype=torch.float32)
  for i, tokens in enumerate(dataset["tokens"]):
    padded[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.float32)

  # Split each sequence into chunks and cache them
  chunks = torch.chunk(padded, chunks=tok_chk_size, dim=1)
  torch.save(chunks, cache_file)
  return chunks

class Decoder(nn.Module):
  # B x T x E input and output, where B is the batch size, T is the token sequence
  # length and E is the embedding dimension. Each token is in a range of [0, V],
  # where V is the vocabulary size
  def __init__(self, E, H, D_k, D_f):
    super().__init__()
    self.H, self.D_k = H, D_k
    self.D_k_scale = 1 / torch.sqrt(self.D_k).item()

    # Linear projection matrices for the query, key and value
    # B x T x E -> B x H x T x D_k, where H is the number of attention heads and
    # D_k is the query, key and value embedding dimension for each attention head
    self.Q_proj = nn.Paramter(torch.tensor(E, D_k * H))
    self.K_proj = nn.Paramter(torch.tensor(E, D_k * H))
    self.V_proj = nn.Paramter(torch.tensor(E, D_k * H))
    # Output projection for the scaled values of the concatenated attention heads
    self.O_proj = nn.Paramter(torch.tensor(D_k * H, E))

    self.norm1 = nn.LayerNorm(E) # After the multihead self-attention
    self.norm2 = nn.LayerNorm(E) # After the feed forward network

    # Feed forward network: B x T x E -> B x T x D_f -> B x T x E,
    # where D_f is the dimension of the hidden layer
    self.ffn = nn.ModuleList([nn.Linear(E, D_f), nn.ReLU(), nn.Linear(D_f, E)])

  def forward(self, x):
    # Linearly project queries, keys and value for each attention head, perform
    # multi-head masked self attention on the input, each concatenate each attention
    # head's values and linearly project them back into embedding space.
    B, T = x.shape[0], x.shape[1]
    Q = (x @ self.Q_proj).reshape(B, T, self.H, self.D_k).transpose(1, 2)
    K = (x @ self.K_proj).reshape(B, T, self.H, self.D_k).transpose(1, 2)
    V = (x @ self.V_proj).reshape(B, T, self.H, self.D_k).transpose(1, 2)

    # TODO: how big are these attention matrices? Their size must cause performance/memory issues...
    scores = (Q @ K.tranpose(-1, -2)) * self.D_k_scale
    # Applying softmax across in the scores matrix of each attention head
    scaled = torch.softmax(scores, dim=-1) @ V
    # Concatenate each attention head and linearly project the result back into embedding space
    attended = scaled.transpose(1, 2).reshape(B, T, self.H * self.D_k) @ self.O_proj

    # Process the attended values
    x = self.norm1(attended + x)
    ffn_output = self.ffn(x)
    return self.norm2(ffn_output + x)

# TODO: Compare model performance across architectural changes in GPT-1, GPT-2, GPT-3
class Transformer(nn.Module):
  # B x T input and output, where B is the batch size, T is the token sequence
  # length. Each token is in range [0, V], where V is the vocabulary size.
  def __init__(self, T, E, H, D_k, D_f, L):
    # Embeddings: B x T -> B x T x E, where E is the embedding dimension
    # Each token selects a row from the learned token embedding matrix
    self.tok_embed = nn.Parameter(torch.tensor(T, E))
    self.pos_embed = nn.Parameter(torch.tensor(T, E))

    # Transformer decoder layers are referred to as decoders because of their
    # auto-regressive masked self-attention, not because of their structure.
    self.layers = nn.ModuleList([Decoder(E, H, D_k, D_f) for _ in range(L)])

  def forward(self, x):
    embeddings = self.tok_embed[x] + self.pos_embed
    for layer in self.layers:
      embeddings = layer(embeddings)

    # Convert embeddings back into tokens by creating a B x T x T logit matrix,
    # getting the most probable token (not using softmax since it won't make a difference)
    # and ... TODO!
    out_logits = embeddings @ self.tok_embed.T
    max_indidces = torch.argmax(out_logits, dim=-1)
    pass

# TODO:
# - implement training loop:
# - add fine details from the original "Attention Is All You Need" and GPT-2 and GPT-3 paper (behind flags for the later models)
# - visualize loss/error, visualize attention scores from each attention head
# - add quantization so a bigger model can fit in kaggle gpu memory
# - read distillation paper

dataset_name = "BabyLM-community/BabyLM-2026-Strict-Small"
tokenizer = tiktoken.encoding_for_model("gpt2")
pad_token = tokenizer.n_vocab # EOS token = Pad token to minimize the vocab size
cache_file = ".cache/prepared-dataset.pth"

try:
  chunks = torch.load(cache_file)
except:
  chunks = prepare_dataset(dataset_name, cache_file, tokenizer, pad_token, 512)

print(len(chunks), chunks[0].shape)
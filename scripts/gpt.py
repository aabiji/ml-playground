import torch
from datasets import load_dataset
from queue import PriorityQueue

seed = 67
torch.manual_seed(seed)

def byte_pair_encode(raw_bytes, id_start, max_rule_size):
  merge_rules = PriorityQueue()
  merge_id = id_start
  data = list(map(lambda x: [x, 1], raw_bytes))

  while merge_rules.qsize() < max_rule_size:
    frequencies, i = {}, 0
    while i < len(data):
      curr_value, curr_skip = data[i]
      if i + curr_skip >= len(data):
          break

      next_value, next_skip = data[i + curr_skip]
      key = (curr_value, next_value)
      if key in frequencies:
          frequencies[key].append(i)
      else:
          frequencies[key] = [i]
      i += curr_skip + next_skip

    most_frequent = max(frequencies.values(), key=lambda a: len(a))
    for idx in most_frequent:
      curr_value, curr_skip = data[idx]
      next_value, next_skip = data[idx + curr_skip]
      priority = -(merge_id - id_start)
      data[idx] = [merge_id, curr_skip + next_skip]
      merge_rules.put((priority, (merge_id, curr_value, next_value)))

    merge_id += 1

  encoded, i = [], 0
  while i < len(data):
    value, skip = data[i]
    encoded.append(value)
    i += skip

  return encoded, merge_rules


def byte_pair_decode(encoded, merge_rules):
  def apply_merge_rule(value, rules):
    if value not in rules:
       return [value]

    expanded = []
    left, right = rules[value]
    expanded.extend(apply_merge_rule(left, rules))
    expanded.extend(apply_merge_rule(right, rules))
    return expanded

  decoded = []
  for value in encoded:
    decoded.extend(apply_merge_rule(value, merge_rules))
  return decoded


dataset = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train")

sample_size = int(0.1 * len(dataset))
sample = dataset.shuffle(seed=seed).select(range(sample_size))
raw_sample = []
for row in sample:
  raw = row["text"].encode("utf-8")
  raw_sample.extend(list(raw))

"""
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




Merge rules: {(iteration, (left, right), replacement)}

Turn list of bytes to a linked list of bytes

Take all rules with the same priority

Scan: Split into groups of 2 bytes. For each group, if it's a merge rule:
- Update pointers (replace next 2 nodes with 1 node)
- Add new left pair and new right pair to a list of affected groups

If the list of affected is empty, do a full scan
Else, iterate through affected only



['h', 'u', 'g', ' ', 'p', 'u', 'g', ' ', 'h', 'u', 'g']
['h', 'H', 'p', 'H', 'h', 'H']
['U', 'p', 'J', 'h', 'H']

[256, 'g', ' ', 'p', 'u', 'g', ' ', 256, 'g']


('h', 'u') -> ('h', 'H')
('g', ' ') -> ('H', ' ')
('p', 'u') -> ('p', 'H')
('h', 'u') -> ('h', 'H')
('g', ' ') -> ('J', ' ')


[
(0, ("u", "g"), H),
(1, ("H", " "), J),
(2, ("h", "u"), U)
]

"""

class ByteNode:
   def __init__(self, n):
      self.n = n
      self.next: None | ByteNode = None
      self.prev: None | ByteNode = None

def compute_merge_pairs(unicode_bytes, vocab_size):
  def add_instance(table, key, value):
    if key in table:
      table[key].append(value)
    else:
      frequencies[key] = [value]

  # Initial frequency pair scan, also building a linked list of the sequence
  head = ByteNode(unicode_bytes[0])
  current = head
  frequencies = {}
  i = 0
  while i < len(unicode_bytes):
    # Last node
    if i + 1 >= len(unicode_bytes):
       node = ByteNode(unicode_bytes[i])
       node.prev = current
       current.next = node
       break

    # Next two nodes
    next2 = ByteNode(unicode_bytes[i + 1])
    next1 = ByteNode(unicode_bytes[i])
    next1.prev = current
    next1.next = next2
    next2.prev = next1
    current.next = next1
    current = next2

    add_instance(frequencies, (unicode_bytes[i], unicode_bytes[i + 1]), next1)
    i += 2

  merge_pairs = {}
  merge_id = 256

  temp = head
  while temp is not None:
    print(temp.n, end=", ")
    temp = temp.next
  print("\n",list(map(lambda x: (x, len(frequencies[x])), frequencies)))

  # Create merge pairs
  while len(merge_pairs.keys()) < vocab_size:
    most_frequent = max(frequencies.keys(), key=lambda k: len(frequencies[k]))
    merge_pairs[most_frequent] = merge_id

    # Update affected pairs
    for ptr in frequencies[most_frequent]:
      left = ptr.prev
      right = ptr.next.next

      replacement = ByteNode(merge_id)
      replacement.next = right
      replacement.prev = left

      if left is not None:
        left.next = replacement
        add_instance(frequencies, (left.n, replacement.n), left)

      if right is not None:
        right.prev = replacement
        add_instance(frequencies, (replacement.n, right.n), replacement)

    merge_id += 1
    del frequencies[most_frequent]

  while head is not None:
    print(head.n, end=", ")
    head = head.next
  print("\n", list(map(lambda x: (x, len(frequencies[x])), frequencies)))
  print(merge_pairs)

compute_merge_pairs([1, 2, 3, 4, 5, 2, 3, 4, 1, 2, 3], 5)

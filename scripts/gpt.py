"""
Instead of having an ensemble of task-specific models and datasets, it would be better to have a
single model that has zero-shot task transfer with as little parameter or architectural changes.

GPT 1:
- It's several layers of transformer decoders followed by a feed forward head.
  - Referred to as a "decoder" because of its masked self-attention, not because of its structure.
- Training is split into two phases: generative pre-training and supervised fine tuning.
  - Pre-training is self-supervised since the likelyhood of predicting the next token is optimized.
  - During post-training, any arbitrary supervised learning objective can be optimized.

GPT 2:
- Maximize p(output | input, task), where the output, input and task are a sequence of symbols.
  Instead of seperating input, task and output, combine them into one input symbol sequence.
- In principle, language modelling should be able to learn various tasks through next token prediction.
  Although the objective is extremely simple, the skills needed to do it effectively are very complex.
  For example, to predict the next word in this sentence: "In french we would say the phrase: 'The dog
  ate my homework' as 'Le chien a mangé mes" as "devoirs", the model needs to have a semantic and
  gramatical understanding of the english sentence and partial french sentence, it needs to recognize
  that this is a translation task, and it needs to understand the nuance between the different
  next word choices, etc.
- Given enough capacity and enough diverse data, both in massive quantities, the model should be able
  to learn abstract representations needed to model language effectively and generalize well to unseen
  examples. The question is *how*.
- Text is tokenized using BPE (on bytes not unicode chars) and merging across punctuation is not
  allowed to enable better use of a limited vocabulary.
"""
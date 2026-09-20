**DUE September 23**

- Write notes about the roformer paper. Publish "Notes on the Rotary Positional Encoding"

- Understand: *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*

- Implement the following improvements to gpt script:
  - Port from jupyter notebook to regular python script. Make GPT its own seperate project.

  - Implement ROPE instead of using learned positional encodings.

  - Implement Linear Attention, that should improve performance and increase the number of steps I can train for.

  - Mask out EOS tokens as well. I wasn't doing that before so the model was treating EOS tokens as valid symbols.

  - Switch to character based tokenization and switch to using Karpathy's Shakespear dataset. The vocabulary size
    should be proportional to the dataset -> large vocab with small dataset size means overfitting on specific tokens
    and terrible performance.

  - Retry the inference demo to see if the changes have made any qualitative differences.

Read these articles/papers:
- Transformer chapter in *Understanding Deep Learning*
- *A Practical Review of mechanistic interpretability for transformer-based language models*
- *Decoupled Weight Decay Regularization*
- *Stochastic Gradient Descent with Warm Restarts*
- *Distilling the Knowledge in a Neural Network*
- *Understanding Deep Learning Requires Re-thinking Generalization*

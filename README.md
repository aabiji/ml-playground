**DUE September 23**

- Read *RoFormer: Enhanced Transformer With Rotary Position Embedding* and compare perf
  against a learned positional embedding

Read these articles/papers:
- [Visual attention variants](https://magazine.sebastianraschka.com/p/visual-attention-variants)
- Transformer chapter in *Understanding Deep Learning*
- Some of Lingling Jin's papers
- *A Practical Review of mechanistic interpretability for transformer-based language models*
- *Decoupled Weight Decay Regularization*
- *Stochastic Gradient Descent with Warm Restarts*
- *Distilling the Knowledge in a Neural Network*
- *Understanding Deep Learning Requires Re-thinking Generalization*

New project:
- Read *Auto-Encoding Variational Bayes*
- Rederive the reparameterization trick myself
- Visualize the 2D latent space
- Interpolate between the two images
- Sample from a prior
- Compare an ordinary auto encoder against a VAE
- Measure reconstruction loss against KL loss
- Deliberately change $\beta$ in $\beta$-VAE
- Observe what happens when the latent dimensionality changes
- Consider a medical anomaly detection demo. Regions with high reconstruction loss should correspond to anomalies.

Project idea:
- Train a model to cancel out human speech for noise cancellation purposes.
- Get a dataset of human speech, and the objective could be minimizing the noise level?
- How is it going to be fast enough for real time cancelling? Going to really need to
  drive down the model latency.
- Speech cancelling should be adaptive to various environments.

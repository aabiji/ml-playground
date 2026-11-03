## TBR

- [x] CTC
- [x] Conformer
- [ ] Fast Conformer
- [ ] ZipFormer
- [ ] Whisper

## ??

- How can we make the model specify who is talking? The model needs to learn differences in pitch?
- Can we simplify the Conformer architecture?
- How does the model handle silence?
- How does the model handle varying noise levels, with or without speech?

## TIL

- Mel spectrograms
  - A signal needs to be sampled at a rate $f_s \geq 2 f_{max}$ in order to avoid distortions.
  - Since most phonetic information lies below 8 Hz, a common sample rate for speech is 16 kHz.
  - Larger STFT windows increase frequency resolution but lower temporal resolution. In other words, a larger STFT window allows for more fine grained frequency bins, but each frame will
    cover a larger time frame, blurring rapid changes. The opposite is also true. Given this tradeoff, a common window size is 25 ms.
  - The x-axis of a MEL spectrogram represents time while its y-axis displays MEL bands. Band group frequencies scale linearly beneath 1000 Hz and logarithmically above, matching how the human ear perceives sound.

- SpecAugment:
  - Time masking is masking out a random vertical strip from $[x, x + k]$. Frequency masking is masking out a random horizontal strip from $[y, y + l]$. Where $k \in [0, K], x \in [0, W - k)$ and $l \in [0, L], y \in [0, H - l)$.
  - `sparse_image_warp`: Compute a thin-plate spline that passes through a random point on the midline, the four corners of the spectrogram and the top and bottom middle points of the spectrogram. Use the thin-plate spline to bilinearly interpolate the spectrogram.
    - To compute the thin-plate spline , a polyharmonic spline is used.
      Suppose there are $M$ control points on a $N \times N$ plane. The goal is to fit a thin-plate spline: a surface that passes through all the control points and is as smooth as possible.
      Each point is defined as the following.

      $$f(x, y) = \sum_{i = 1}^{M} w_i \phi(r) + p(x, y)$$

      Where $r = \sqrt{(x - c_{ix})^2 + (y - c_{iy})^2}$ is the distance between the point and the specific control point,
      $w_i$ is a weight for the radial basis function $\phi$ that satisfies the following orthogonality constraints:
      $\sum_{i = 1}^{N} w_i = \sum_{i = 1}^{N} w_i \c_i = 0$
      and $p(x, y) = a + bx + cy$ is a polynomial function that specifies the background shape of the thin-plate spine.
      To solve for $w$ and $a, b, c$, the equation above can be reworked into a system of linear equations.

      $$
      \begin{bmatrix} K & P \\ P^T & 0 \end{bmatrix}
      \begin{bmatrix} \boldsymbol{w} \\ \boldsymbol{c} \end{bmatrix}
      = \begin{bmatrix} \boldsymbol{z} \\ 0 \end{bmatrix}
      $$

      $K \in R^{M \times M}$ and each $K_{ij}$ is the radial basis function evaluated at the distance between the control points $i$ and $j$.
      $P \in R^{M \times 3}$ and each $P_{i} = \begin{bmatrix} 1 & x & y \end{bmatrix}$ and $\boldsymbol{z} \in R^{M \times 1}$ is the value of each control point.

      $\boldsymbol{w} \in R^{M \times 1}$ and $\boldsymbol{c} = \begin{bmatrix} a \\ b \\ c \end{bmatrix}$ are the weights and polynomial coefficients that are to be solved for.

    - For every spectrogram in a batch, for every batch, for every epoch, a pipeline might look like the following.
      1. Solve for w and c
      2. Broadcast weights: $M x 1 \rightarrow M x N x N$
      3. Compute polynomial for each position $N x N$
      4. Compute $\phi$ for each position and weight into an $M x N x N$ tensor, multiply by broadcasted weights, sum across M to get a $N x N$ matrix, add polynomial
      5. Then use the resulting thin-plate spin to bilinearly interpolate each pixel on the souce image
         So, the tiny regularization benefits derived from doing time warping is just not worth its comical inefficiency.

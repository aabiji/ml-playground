**Mel spectrograms**

- A signal needs to be sampled at a rate $f_s \geq 2 f_{max}$ in order to avoid distortions.
- Since most phonetic information lies below 8 Hz, a common sample rate for speech is 16 kHz.
- Larger STFT windows increase frequency resolution but lower temporal resolution. In other words, a larger STFT window allows for more fine grained frequency bins, but each frame will
  cover a larger time frame, blurring rapid changes. The opposite is also true. Given this tradeoff, a common window size is 25 ms.
- The x-axis of a MEL spectrogram represents time while its y-axis displays MEL bands. Band group frequencies scale linearly beneath 1000 Hz and logarithmically above, matching how the human ear perceives sound.

**SpecAugment**

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

**CTC**

- A speech recognition model takes audio $X$, splits it into $T$ frames and at each frame outputs a probability distribution $p_t(\cdot | X)$ over a vocabulary $V$ that includes a special blank token $\epsilon$. The goal is to produce a transcript $Y = \{ y_1, y_2, ... y_N \}$. The problem is that we don't know which frames correspond to which tokens. Training with ordinary cross-entropy would require a label for each frame, which is normally not present. Phoneme duration varies with the speaker's speed, emotion and accent, so the model may emit the same token over several consecutive frames. Many frame-level token sequences, which are referred to as alignments, can therefore correspond to the same transcript when a function $B$ merges consecutively repeated tokens and removes blanks.

- CTC maximizes the probability of the transcript by summing the probability of each token over time, for all possible valid alignments. The loss is $-\log p(Y | X)$ and the product over $t$ is valid because CTC asummes frame outputs are conditionally independent from $X$.

$$p(Y | X) = \sum_{A \in A_{X, Y}} \prod_{t = 1}^{T} p_t(a_t | x_t) \qquad A_{X, Y} = \{ A \in V^T : B(A) = Y\}$$

- Evaluating all $V^T$ possible alignments, most of which are invalid, is infeasible. However, this problem can be reformulated as grid traversal problem. Let's define an $M \times T$ grid, where $Z$ is the output sequence $Y$ padded with blank tokens.

$$Z \in \{ \epsilon, y_1, \epsilon, y_2, \epsilon, ... , \epsilon, y_N, \epsilon \}, \qquad M = 2 |Y| + 1$$

Each cell in the grid contains $\alpha$, the total probability of all paths that start on $(t, z_1)$, end on $(t, z_m)$, sit on one position each frame and always step forward with time. All cells will be initialized to zero. The goal is to compute the total probability of all paths that produce a valid alignment.

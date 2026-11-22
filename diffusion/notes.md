https://www.youtube.com/watch?v=wMmqCMwuM2Qs
https://friedmanroy.github.io/blog/2024/gen0/
https://friedmanroy.github.io/blog/2022/Langevin/

Mini goal: work through [denoising score matching](https://yang-song.net/blog/2021/score/) as a way to prepare for the ddpm paper
- Sample input values and compute score for a known data distribution - probably some mixture of gaussians that
  I made up. Visualize score as a vector field.
- Train a score model by minimizing Fisher divergence. I should be getting really low error between the learned score and the actual score.
- Sample the data distribution using Langevin dynamics
- Read the rest of the article for next steps

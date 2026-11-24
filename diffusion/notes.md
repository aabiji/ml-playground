https://www.youtube.com/watch?v=wMmqCMwuM2Qs
https://friedmanroy.github.io/blog/2022/Langevin/
https://yang-song.net/blog/2021/score/

We assume that samples are i.i.d from a distribution we don't have the analytic form for.
If we can estimate the distribution from our input samples, we can use our estimated
distribution to sample novel data, and determine the probability of samples being in the distribution.

**Interesting applications**:
- anomaly detection -> if an input has a low probability of being sampled from the distribution it must be a fake!
- image, video, audio generation (as a form of augmentation, as a product, etc)

**Probability**:
A probability is a function that satsifies Kolmogorov's axioms:
- P(A) >= 0
- P(everything that could happen) = 1
- For disjoint events, P(A or B) = P(A) + P(B)
Probabilities take sets as inputs

A probability density function for a random variable X is a function p(x) such that
- p(x) >= 0 for all x
- \int_{-inf}^{inf} p(x)dx = 1 over
- P(X \in A) = \int_A p(x) dx
p(x) is NOT a probability, it is a value that is proportional to how likely a sample is to land in a small volume of a fixed size around x
p(x_1) > p(x_2) tells us that samples are more likely to land in a small window around x_1 rather than in an equally sized window around x_2

**Questions**:
- Can we control what we sample when using Annealed Langevin?

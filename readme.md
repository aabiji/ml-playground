Ideas:

- Read Understanding Deep Learning to learn more

- Implement [RMSNorm and LayerScale](https://arxiv.org/pdf/2602.08071) and [2D Rope](https://arxiv.org/pdf/2403.13298) on a ViT model and train it on an interesting dataset
  - Visualize the [cls] row of the attention matrix overlaid on top of an image to visualize how each layer of a small transformer learns what to pay attention to over the training schedule.
  - https://blog.trainindata.com/the-ultimate-guide-to-deep-learning-hyperparameter-tuning/
  - https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
  - Use the [Car-1000 dataset](https://github.com/toggle1995/Car-1000)

- Explore ASR
  - Distill the Nemotron model into a 100 M or lower parameter model
    - Read [Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531)
    - Read [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102)
    - Read [Stateful Conformer With Cache-Based Inference For Streaming Automatic Speech Recognition](https://arxiv.org/pdf/2312.17279)
    - Read [Fast Conformer With Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/pdf/2305.05084)
    - Inspect the teacher's architecture, tokenizer, output format, training framework
      and available training checkpoints
    - Train a tiner clone of the model using the 600M param teacher mode

  - Write a simple real time voice dictation script using the distilled model
    - Stream audio from microphone
    - Run the model on the streamed audio and convert text into windows key presses
    - Package the script as a daemon that runs in the background as admin
    - Add a simple taskbar applet to toggle it on/off

- Implement the DDPM paper
 - Pros: Entirely new concept, lots of math. Interesting demo.
 - Cons: Likely to be another 100 line project, would only keep me occupied for a week, not sure how to extend it.

- Pose estimation to 3D model movement
  - Implement a pose estimation model.
  - Map the estimated poses to a 3D skeletal animation file.
  - Implement a basic skeletal animation demo in C++, compile to WASM for web demo.
  - Pros: Really cool demo.
  - Cons: Old idea I've tried multiple times with no success. No idea where to even start on the model.
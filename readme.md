Ideas:

- Read and implement the DDPM paper

- Pose estimation to 3D model movement
  - Implement a pose estimation model.
    - Read *Stacked Hourglass Networks for Human Pose Estimation* and implement a simple hourglass
    - Read *Searching for MobileNetV3* and take a pre-trained MobileNetV3 backbone, strip off the
      classification head and add a fully connected layer that directly outputs xy for 17 keypoints
    - Read *BlazePose: On device Real-time Body Pose Tracking* and implement and train the BlazePose architecture
  - Map the estimated poses to a 3D skeletal animation file.
  - Implement a basic skeletal animation demo in C++, compile to WASM for web demo.

- Read *Understanding Deep Learning* and do the practice assignments to learn more

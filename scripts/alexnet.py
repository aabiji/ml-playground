# The following is a "from scratch" reimplementation of the AlexNet paper.
# It's not an identical replication, since this model trains on a much smaller
# dataset, and only outputs a distribution over 9 class labels. Additionally,
# PCA is not used for data augmentation, the learning stays fixed and only 3
# image crops are used to augment randomly selecting images.

import numpy as np
import os
import random
from PIL import Image, ImageOps
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


def randomly_select_files(outer_folder_path, num_files):
  class_folders = Path(outer_folder_path).iterdir()
  all_files = []

  for folder in class_folders:
    with os.scandir(folder.resolve()) as entries:
      all_files.extend([entry.path for entry in entries if entry.is_file()])

  selection_count = min(len(all_files), num_files)
  return random.sample(all_files, selection_count)


def load_dataset(outer_folder_path, classes, num_crops, num_batches, batch_size):
  paths = randomly_select_files(outer_folder_path, num_batches * batch_size)

  total = batch_size * num_crops
  imgs = np.zeros((num_batches, total, 3, 227, 227))
  labels = np.zeros((num_batches, total, len(classes), 1))

  for i in range(num_batches):
    for j in range(batch_size):
      path = paths[i * batch_size + j]

      name = Path(path).parent.name
      one_hot = np.zeros((len(classes), 1))
      one_hot[classes[name]] = 1

      file = Image.open(path)
      resized = ImageOps.pad(file, (256, 256), color="black",
        centering=(0, 0), method=Image.Resampling.NEAREST)

      # Augment the data by using randomly cropped sections of the image
      for k in range(num_crops):
        x, y = np.random.randint(29), np.random.randint(29)
        crop = resized.crop((x, y, x + 227, y + 227))
        arr = np.array(crop).astype(np.float32) / 255.0
        imgs[i, j * num_crops + k] = arr.T
        labels[i, j * num_crops + k] = one_hot

      file.close()

  return imgs, labels


def expand(data, pad, dilation):
  output = data

  # Dilate - Insert a number of zeroes between
  # elements in the array in the specific axes
  if dilation > 1:
    new_shape = list(data.shape)
    slices = [slice(None) for _ in range(data.ndim)]
    last = len(data.shape) - 1

    # Update the sizes of the last two axes
    new_shape[last] = \
      data.shape[last] + (data.shape[last] - 1) * (dilation - 1)
    new_shape[last - 1] = \
      data.shape[last - 1] + (data.shape[last - 1] - 1) * (dilation - 1)

    slices[last] = slice(None, None, dilation)
    slices[last - 1] = slice(None, None, dilation)

    output = np.zeros(new_shape, dtype=data.dtype)
    output[tuple(slices)] = data

  # Pad - Only pad the last two dimensions
  if pad > 0:
    dims = [(0, 0) for _ in range(len(data.shape))]
    dims[-1] = dims[-2] = (pad, pad)
    output = np.pad(output, pad_width=tuple(dims),
                    mode="constant", constant_values=0)

  return output


def average(data):
  return data.sum(axis=0) / data.shape[0] # Average values across a batch


# Dimensions:
# B = batch size, H = height, W = width, C = number of channels
# C_i = number of input channels (convolutional) or previous layer width (linear)
# C_o = number of output channels (convolutional) or next layer width (linear)
# K = kernel size
#
# layer shape = (B, C, H, W), kernel shape = (C_o, C_i, K, K)
def cnn_layer_shape(layer_shape, kernel_shape, stride, pad):
  return (
    kernel_shape[0],
    int((layer_shape[-2] + 2 * pad - kernel_shape[-1]) / stride) + 1,
    int((layer_shape[-1] + 2 * pad - kernel_shape[-1]) / stride) + 1
  )


def convolution(layer, kernel, stride, layer_pad,
                layer_dilation, kernel_pad, kernel_dilation):
  layer = expand(layer, layer_pad, layer_dilation) # (B, C_i, H, W)

  # foward pass: (C_o, C_i, K, K), backward pass: (C_o, K, K)
  kernel = expand(kernel, kernel_pad, kernel_dilation)

  axes = (layer.ndim - 2, layer.ndim - 1)
  kernel_shape = (kernel.shape[-2], kernel.shape[-1])

  regions = sliding_window_view(layer, kernel_shape, axes)
  if layer.ndim == 6:
    # forward pass: (B, C_i, H, W, K, K)
    regions = regions[:, :, ::stride, ::stride, :, :]
  else:
    # backward pass: (C_i, H, W, K, K)
    regions = regions[:, :, ::stride, ::stride]

  if kernel.ndim == 3:
    # weight gradient: (C_o, H, W)
    result = np.einsum("bihwyx,oyx->boihw", regions, kernel)
    return average(result)

  if layer_pad == kernel.shape[-1] - 1:
    # layer gradient: (C_i, C_o, H, W)
    return np.einsum("ohwyx,ioyx->ihw", regions, kernel)

  # standard convolution: (B, C_o, H, W)
  return np.einsum("bihwyx,oiyx->bohw", regions, kernel)


# See section 3.3 of the paper
def local_response_normalization(conv_layer):
  b, channels, h, w = conv_layer.shape
  normalized = np.zeros((b, channels, h, w))

  for i in range(channels):
    a, b = max(0, i - 2), min(channels - 1, i + 3)
    neighbors = np.square(conv_layer[:, a:b]).sum(axis=1)
    normalized[:, i] = np.pow(2 + 10e-4 * neighbors, 0.75)

  return normalized


def initialize_params(shape, one_bias):
  # See section 5 of the paper
  rng = np.random.default_rng()
  weight = rng.normal(0.0, 0.01, shape)
  bias = np.ones((shape[0], 1)) if one_bias else np.zeros((shape[0], 1))
  return weight, bias


def softmax(data):
  e = np.exp(data - np.max(data))
  return e / np.sum(e)


class Linear:
  def __init__(self, weights_shape, one_bias, dropout):
    self.weights, self.biases = initialize_params(weights_shape, one_bias)
    self.dropout = dropout
    self.data = None
    self.weights_gradient = None
    self.biases_gradient = None

  def forward(self, prev_layer, test=False):
    self.data = self.biases + self.weights @ prev_layer

    if self.dropout:
      mask = 0.5 if test else np.random.randint(2, size=self.data.shape)
      self.data *= mask

  def backward(self, prev_layer, gradient):
    self.biases_gradient = np.sum(gradient, axis=0)
    self.weights_gradient = gradient @ np.swapaxes(prev_layer, -1, -2)
    self.weights_gradient = average(self.weights_gradient)
    return self.weights.T @ gradient


class Convolutional:
  def __init__(self, weights_shape, stride, padding, one_bias, normalize):
    self.weights, self.biases = initialize_params(weights_shape, one_bias)
    self.stride = stride
    self.padding = padding
    self.normalize = normalize
    self.data = None
    self.weights_gradient = None
    self.biases_gradient = None

  def forward(self, prev_layer):
    output = convolution(prev_layer, self.weights,
                         self.stride, self.padding, 1, 0, 1)
    self.data = self.biases.squeeze()[:, None, None] + output
    if self.normalize:
      self.data = local_response_normalization(self.data)

  def backward(self, prev_layer, gradient):
    self.biases_gradient = gradient.sum(axis=(1, 2)).reshape(-1, 1)
    self.weights_gradient = convolution(prev_layer, gradient, 1,
                                        self.padding, 1, 0, self.stride)

    # Turn (C_o, C_i, K, K) ->  (C_i, C_o, K, K) and flip the kernel contents
    transposed = np.transpose(self.weights, (1, 0, 2, 3))
    rotated = np.rot90(transposed, k=2, axes=(2, 3))
    kernel_size = self.weights.shape[-1]

    gradient = convolution(gradient, rotated, 1, kernel_size - 1, self.stride, 0, 1)
    if self.padding > 0: # Strip padding
      gradient = gradient[:, self.padding:-self.padding, self.padding:-self.padding]
    return gradient


class Maxpool:
  def __init__(self, kernel_size, stride, flatten=False):
    self.data = None
    self.argmax = None
    self.old_shape = None
    self.new_shape = None
    self.N = kernel_size
    self.S = stride
    self.flatten = flatten

  def forward(self, prev_layer):
    axes = (prev_layer.ndim - 2, prev_layer.ndim - 1)

    # Extract max value from each (N, N) regions, shape: (B, C, H, W, K, K)
    regions = sliding_window_view(prev_layer, (self.N, self.N), axes)
    regions = regions[:, :, ::self.S, ::self.S]
    self.data = np.max(regions, axis=(-2, -1))
    self.old_shape = prev_layer.shape
    self.new_shape = self.data.shape

    # Each value in (B, C, H, W) stores an index from 0 to (N * N - 1)
    b, c, h, w, _, _ = regions.shape
    self.argmax = np.reshape(regions, (b, c, h, w, self.N * self.N)).argmax(axis=-1)

    if self.flatten: # Turn into a column vector while preserving batch dimension
      self.data = self.data.reshape(self.data.shape[0], -1, 1)

  def backward(self, prev_layer, gradient):
    # The derivative of a max pool is a max unpool
    if self.flatten:
      self.data = self.data.reshape(self.new_shape)

    i = self.argmax // self.N
    j = self.argmax % self.N
    b, c, h, w = np.indices(self.data.shape)

    # Assign the max values to their original positions
    gradient = np.zeros(self.old_shape)
    gradient[b, c, h * self.S + i, w * self.S + j] = self.data
    return average(gradient) # Removes the batch dimension


class ReLU:
  def __init__(self):
    self.data = None

  def forward(self, prev_layer):
    self.data = prev_layer.clip(0.0)

  def backward(self, prev_layer, gradient):
    return average((prev_layer > 0) * gradient)


classes = {
  "angry": 0, "confused": 1, "disgust": 2,
  "fear":  3, "happy":    4, "neutral": 5,
  "sad":   6, "shy":     7, "surprise": 8
}

#epochs, num_batches, B, num_crops = 90, 128, 132, 5
epochs, num_batches, B, num_crops = 1, 5, 5, 1
learning_rate = 0.01
num_layers = 8 # Excluding maxpool and relu layers

imgs, labels = load_dataset(
  "../data/fane/fane_data/", classes, num_crops, num_batches + 1, B)
train_imgs, train_labels = imgs[1:], labels[1:]
test_imgs, test_labels = imgs[0], labels[0]

layer_info = [
  # Convolutional layers:
  # ((B, C_o, C_i, K, K), stride, padding, one_bias, normalize)
  ((96, 3, 11,  11), 4, 0, False, True),
  ((256, 96,  5, 5), 1, 2, True,  True),
  ((384, 256, 3, 3), 1, 1, False, False),
  ((384, 384, 3, 3), 1, 1, True,  False),
  ((256, 384, 3, 3), 1, 1, True,  False),

  # Linear layers:
  # ((C_o, C_i), one_bias, dropout)
  ((4096, 9216), True, True),
  ((4096, 4096), True, True),
  ((9,    4096), True, False)
]

layers = [
  Convolutional(*layer_info[0]), ReLU(), Maxpool(3, 2),
  Convolutional(*layer_info[1]), ReLU(), Maxpool(3, 2),
  Convolutional(*layer_info[2]), ReLU(),
  Convolutional(*layer_info[3]), ReLU(),
  Convolutional(*layer_info[4]), ReLU(), Maxpool(3, 2, True),
  Linear(*layer_info[5]), ReLU(),
  Linear(*layer_info[6]), ReLU(),
  Linear(*layer_info[7])
]

wmomentum = [np.zeros(layer_info[i][0]) for i in range(num_layers)]
bmomentum = [np.zeros((layer_info[i][0][0], 1)) for i in range(num_layers)]

# Train
for epoch in range(epochs):
  for b in range(num_batches):
    print(f"Epoch {epoch + 1}, Batch {b + 1}/{num_batches}")

    batch_x, batch_y = train_imgs[b], train_labels[b]
    wgradients, bgradients = [0] * num_layers, [0] * num_layers

    # Forward propagation
    for i, layer in enumerate(layers):
      prev_layer = batch_x if i == 0 else layers[i - 1].data
      layer.forward(prev_layer)

    output = softmax(layers[-1].data)
    gradient = output - batch_y # Derivative of cross entropy

    # Backward propagation
    count = num_layers - 1
    for i in range(len(layers) - 1, -1, -1):
      prev_layer = batch_x if i == 0 else layers[i - 1].data
      gradient = layers[i].backward(prev_layer, gradient)

      if type(layers[i]) is Linear or type(layers[i]) is Convolutional:
        wgradients[count] = layers[i].weights_gradient
        bgradients[count] = layers[i].biases_gradient
        count -= 1

    indices = [0, 3, 6, 8, 10, 13, 15, 17, 19]
    for j in range(num_layers):
      i = indices[j]
      wmomentum[j] = 0.9 * wmomentum[j] - \
                     0.0005 * learning_rate * layers[i].weights - \
                     learning_rate * wgradients[j]
      layers[i].weights += wmomentum[j]

      bmomentum[j] = 0.9 * bmomentum[j] - learning_rate * bgradients[j]
      layers[i].biases += bmomentum[j]


# Inference
for i, layer in enumerate(layers):
  prev_layer = test_imgs if i == 0 else layers[i - 1].data
  layer.forward(prev_layer)

# Average across all image crops
output = softmax(layers[-1].data)
num_images = output.shape[0] // num_crops
output = output.reshape(num_images, num_crops, -1).mean(axis=1)

max_idx = np.argmax(output, axis=1)
one_hot = np.zeros((B, len(classes), 1))
one_hot[np.arange(len(max_idx)), max_idx] = 1

num_correct = (one_hot == test_labels).all(axis=1).sum()
accuracy = 100 * num_correct / (B * num_crops)
print(f"Test accuracy: {accuracy}%")

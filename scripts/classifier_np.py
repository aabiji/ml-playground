import numpy as np
import matplotlib.pyplot as plt
import local_datasets

def init_network(input_dim, output_dim, hidden_dim, depth):
    bias   = [np.zeros((hidden_dim, 1)) for _ in range(depth)]
    bias[-1] = np.zeros(( output_dim, 1))

    weights = [np.zeros((hidden_dim, hidden_dim)) for _ in range(depth)]
    weights[0] = np.zeros((hidden_dim, input_dim))
    weights[-1] = np.zeros((output_dim, hidden_dim))

    # He initialize the weights
    rng = np.random.default_rng()
    for i in range(depth):
        variance = 2.0 / hidden_dim
        if i == 0:
            variance = 4.0 / (input_dim + hidden_dim)
        elif i == depth - 1:
            variance = 4.0 / (hidden_dim + output_dim)
        weights[i] = rng.normal(0, np.sqrt(variance), size=weights[i].shape)

    return weights, bias

def forward(weights, bias, in_sample, depth, hidden_dim, out_shape):
    # Input layer, hidden layers, output layer
    layers = [np.zeros((hidden_dim, 1)) for _ in range(depth + 1)]
    layers[0] = in_sample

    # Linear combination and ReLU
    for i in range(1, depth + 1):
        layers[i] = bias[i - 1] + weights[i - 1] @ layers[i - 1]
        if i < depth: # Only apply ReLU to hidden lyaers
            layers[i] = layers[i].clip(0.0)

    # Apply softmax to the network output
    output = np.reshape(layers[-1], out_shape)
    max_logit = np.max(output, axis=1)
    e = np.exp(output - max_logit)
    layers[-1] = e / e.sum(axis=1)
    return layers

def backpropagation(expected_output, layers, weights, depth):
    dl_dweights = [np.array(0) for _ in range(depth)]
    dl_dbias    = [np.array(0) for _ in range(depth)]

    # Start with the derivative of the loss with respect to the output layer
    global_grad = 2.0 * (layers[-1] - expected_output).T

    # Compute the local gradient, update the global gradient and pass the
    # global gradient to previous layers, from right to left
    for i in range(depth - 1, -1, -1):
        dl_dbias[i] = global_grad.copy()
        dl_dweights[i] = np.outer(global_grad, layers[i].T)
        if i > 0:
            global_grad = weights[i].T @ global_grad
            # Same thing as element-wise multiplication of ReLU derivative
            # Derivatives are always for the hidden layers, not the input layer
            global_grad = np.where(layers[i] < 0, 0, global_grad)

    return dl_dweights, dl_dbias

def adam(m, v, t, beta1, beta2, alpha, depth, params, gradients):
    for i in range(depth):
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]
        v[i] = beta2 * v[i] + (1 - beta2) * np.square(gradients[i])

        amplified_m = m[i] / (1 - np.pow(beta1, t + 1))
        amplified_v = v[i] / (1 - np.pow(beta2, t + 1))
        params[i] -= alpha * amplified_m / (np.sqrt(amplified_v) + 0.001)

#x_train, y_train, x_test, y_test = local_datasets.to_nparray(*local_datasets.load_iris())
x_train, y_train, x_test, y_test = local_datasets.to_nparray(*local_datasets.load_mnist(
    "../data/mnist/train-images.idx3-ubyte",
    "../data/mnist/train-labels.idx1-ubyte",
    "../data/mnist/t10k-images.idx3-ubyte",
    "../data/mnist/t10k-labels.idx1-ubyte"
))
input_dim, output_dim, hidden_dim, depth = x_train.shape[1], y_train.shape[1], 10, 3
weights, bias = init_network(input_dim, output_dim, hidden_dim, depth)

beta1, beta2, learning_rate = 0.9, 0.999, 0.001
num_batches, num_epochs = 60, 150
batch_size = int(x_train.shape[0] / num_batches)
in_shape, out_shape = (x_train.shape[1], 1), (1, y_train.shape[1])

# Momentum and squared momentum used in Adam
m_weights = [np.zeros(weights[i].shape) for i in range(depth)]
v_weights = [np.zeros(weights[i].shape) for i in range(depth)]
m_bias = [np.zeros(bias[i].shape) for i in range(depth)]
v_bias = [np.zeros(bias[i].shape) for i in range(depth)]

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Epoch")
ax.set_xlim(0, num_epochs)
ax.set_ylabel("Accuracy (%)")

accuracy_line, = ax.plot([], [], color="blue")
accuracy_line.set_xdata(np.arange(0, num_epochs, 1))
accuracies = [0.0 for _ in range(num_epochs)]

for epoch in range(num_epochs):
    for _ in range(num_batches):
        # Sample a batch
        rng = np.random.default_rng()
        indices = rng.integers(0, x_train.shape[0], batch_size)

        weight_gradient = [np.zeros_like(weights[i]) for i in range(depth)]
        bias_gradient = [np.zeros_like(bias[i]) for i in range(depth)]

        # Model training...
        for batch_index in indices:
            x = np.reshape(x_train[batch_index], in_shape)
            y = np.reshape(y_train[batch_index], out_shape)
            layers = forward(weights, bias, x, depth, hidden_dim, out_shape)

            # Accumulate gradients for each layer
            dl_dweights, dl_dbias = backpropagation(y, layers, weights, depth)
            for layer in range(depth):
                weight_gradient[layer] += dl_dweights[layer]
                bias_gradient[layer] += dl_dbias[layer]

        # Adam optimization for weights and biases
        avg_wgradient = [x / float(batch_size) for x in weight_gradient]
        avg_bgradient = [x / float(batch_size) for x in bias_gradient]
        adam(m_weights, v_weights, epoch, beta1, beta2, learning_rate, depth, weights, avg_wgradient)
        adam(m_bias, v_bias, epoch, beta1, beta2, learning_rate, depth, bias, avg_bgradient)

    # Run inference on the test sample to measure accuracy
    num_correct = 0
    for sample_idx in range(x_test.shape[0]):
        x_sample = np.reshape(x_test[sample_idx], in_shape)
        layers = forward(weights, bias, x_sample, depth, hidden_dim, out_shape)

        one_hot_prediction = np.zeros(out_shape[1])
        one_hot_prediction[np.argmax(layers[-1])] = 1
        true_output = np.squeeze(y_test[sample_idx])

        num_correct += 1 if (one_hot_prediction == true_output).all() else 0

    accuracies[epoch] = 100.0 * num_correct / x_test.shape[0]
    accuracy_line.set_ydata(accuracies)
    ax.relim()
    ax.autoscale_view()
    print(f"Epoch: {epoch + 1}/{num_epochs} | Accuracy: {accuracies[epoch]}%")
    plt.pause(0.01)

plt.annotate(f"{accuracies[-1]:.2f}", xy=(num_epochs - 1, accuracies[-1]),
             xytext=(10, 0), textcoords="offset points", va="center")
plt.ioff()
plt.show()
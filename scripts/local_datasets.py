import struct, warnings, pickle, tarfile, requests
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

def to_nparray(x_train, y_train, x_test, y_test):
    return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()

def load_iris():
    df = pd.read_csv("../data/IRIS.csv")
    df = pd.get_dummies(df, columns=["species"], dtype=int)
    df = df.sample(frac = 1)

    in_columns = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    in_values = torch.tensor(in_columns.values, dtype=torch.float32)
    x_train, x_test = in_values[0:100], in_values[100:]

    out_columns = df[["species_Iris-setosa", "species_Iris-versicolor", "species_Iris-virginica"]]
    out_values = torch.tensor(out_columns.values, dtype=torch.float32)
    y_train, y_test = out_values[0:100], out_values[100:]
    return x_train, y_train, x_test, y_test

def load_mnist(train_input_path, train_label_path, test_input_path, test_label_path):
    inputs, labels = [], []

    for path in [train_input_path, test_input_path]:
        with open(path, "rb") as file:
            _, size = struct.unpack(">II", file.read(8))
            height, width = struct.unpack(">II", file.read(8))
            buf = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
            buf = np.reshape(buf, (size, height * width))
            tensor = torch.tensor(buf, dtype=torch.float32)
            inputs.append(tensor / 255.0)

    for path in [train_label_path, test_label_path]:
        with open(path, "rb") as file:
            _, size = struct.unpack(">II", file.read(8))
            buf = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
            tensor = torch.tensor(buf, dtype=torch.long)
            one_hot = F.one_hot(tensor, num_classes=10)
            labels.append(one_hot.float())

    return inputs[0], labels[0], inputs[1], labels[1]

def download_dataset():
    path = "cifar-10-python.tar.gz"
    if Path(path).is_file():
        print("Dataset already downloaded")
        return

    url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
    response = requests.get(url)
    assert response.status_code == 200
    with open(path, "wb") as file:
        file.write(response.content)

    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=".", filter="data")

    print("Downloaded and extracted dataset.")


def load_cifar10(device):
    download_dataset()
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*align should be passed as Python or NumPy boolean.*"
    )

    classes = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]
    paths = [f"cifar-10-batches-py/data_batch_{i + 1}" for i in range(5)]
    paths.append(f"cifar-10-batches-py/test_batch")
    batches = []

    for path in paths:
        with open(path, "rb") as file:
            dict = pickle.load(file, encoding="bytes")
            data, labels = dict[b"data"], dict[b"labels"]
            images = np.zeros((10000, 3, 32, 32))
            images[:, 0] = data[:, :1024].reshape(10000, 32, 32) / 255
            images[:, 1] = data[:, 1024:2048].reshape(10000, 32, 32) / 255
            images[:, 2] = data[:, 2048:].reshape(10000, 32, 32) / 255
            batch = (
                torch.tensor(images, dtype=torch.float32).to(device),
                torch.tensor(labels, dtype=torch.int64).to(device))
            batches.append(batch)

    return batches, classes

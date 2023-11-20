import numpy
import torch
import pandas as pd
import torchaudio
from sklearn import datasets
from sklearn.model_selection import train_test_split

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, data, seq_len=20):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.__len__() - self.seq_len - 1

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_len], self.data[index+self.seq_len])


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, n, k, num_samples):
        self.n = n
        self.k = k
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        signal = torch.distributions.uniform.Uniform(-1, 1).sample(torch.Size([self.n]))
        return (signal, signal[self.n-self.k])

class Iris(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        iris_dataset = datasets.load_iris()
        data = iris_dataset.data.astype(numpy.float32)
        target = iris_dataset.target.astype(numpy.int64)
        train_X, test_X, train_Y, test_Y = train_test_split(data, target, test_size=0.2)
        if is_train:
            self.X = train_X
            self.Y = train_Y
        else:
            self.X = test_X
            self.Y = test_Y
        self.words = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class SpeechCommands(torch.utils.data.Dataset):
    def __init__(self, max_sequence_length, is_train=True):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS("./", download=True, subset="training" if is_train else "validation")
        self.max_sequence_length = max_sequence_length
        self.words = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "five",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]
        self.length = min(1024, len(self.dataset))
        self.scale = len(self.dataset) / self.length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = int(self.scale * index)
        x, _, y, _, _ = self.dataset[index]
        x = x[0, :self.max_sequence_length]
        y = torch.tensor(self.words.index(y), dtype=torch.int64)
        return x, y

def load_csv(dataset_name: str, seq_len=20, val_size: float = 0.33, sigma_magnitude=255.0):
    if dataset_name not in ["SP500", "airline-passengers"]:
        raise Exception("CSV dataset name expected to be one of SP500 and airline-passengers, got", repr(dataset_name))
    file_path = f"data/{dataset_name}.csv"

    df = pd.read_csv(file_path)
    df["data"] = df["data"].astype(float)
    data = torch.tensor(df["data"].to_list())
    train_data = data[0 : int(len(data) * (1 - val_size))]
    val_data = data[int(len(data) * (1 - val_size)) :]

    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std * sigma_magnitude
    val_data = (val_data - train_mean) / train_std * sigma_magnitude

    train_dataset = TimeseriesDataset(train_data, seq_len)
    val_dataset = TimeseriesDataset(val_data, seq_len)

    return train_dataset, val_dataset

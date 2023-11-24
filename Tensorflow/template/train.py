from typing import Literal
import pandas as pd
import tensorflow as tf
from template.dataset.classification_dataset import ClassificationDataset
from template.dataset.memory import MemoryDataset
from template.dataset.window_generator import WindowGenerator
from template.model import get_classification_model, get_model


def prepare_model(
    dataset_name: Literal["speech_commands"] | Literal["iris"] | Literal["SP500"] | Literal["airline-passengers"] | Literal["memory"],
    activation: Literal["relu"] | Literal["tanh"] | Literal["custom"] = "relu",
    hidden_size: int = 10,
    sequence_length: int = 20,
    custom_max_value: float = 255.0,
    sigma_magnitude: float = 1.0,
    signal_length: int = 100,
    n: int = 10,
    k: int = 1,
    num_samples: int = 1000,
):
    if dataset_name in ["SP500", "airline-passengers"]:
        df = pd.read_csv(f"data/{dataset_name}.csv")
        df["data"] = df["data"].astype(float)

        dataset = WindowGenerator(
            input_width=sequence_length,
            label_width=1,
            shift=1,
            df=df,
            label_columns=["data"],
            normalize=True,
            sigma_magnitude=sigma_magnitude,
        )
        model = get_model(hidden_size, 1, input_shape=[sequence_length, 1], activation=activation, custom_max_value=custom_max_value)
    elif dataset_name in ["iris", "speech_commands"]:
        dataset = ClassificationDataset(dataset_name=dataset_name, max_sequence_length=sequence_length)
        model = get_classification_model(hidden_size, dataset.classes, input_shape=[dataset.signal_len, 1], activation=activation, custom_max_value=1.0)
    else:
        dataset = MemoryDataset(num_samples=num_samples, n=n, k=k)
        model = get_model(hidden_size, 1, input_shape=[n, 1], activation=activation, custom_max_value=1.0)
    print(model.summary())
    model.compile(
        loss=dataset.loss,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[dataset.metrics],
    )
    
    return dataset, model

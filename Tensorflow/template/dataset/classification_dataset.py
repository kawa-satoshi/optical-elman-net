from typing import Literal
import tensorflow as tf
import tensorflow_datasets as tfds

class ClassificationDataset:
    def __init__(
        self,
        dataset_name: Literal["speech_commands"] | Literal["iris"],
        max_sequence_length = 1600,
        batch_size = 32,
    ):
        self.train = tfds.load(dataset_name, split='train', as_supervised=True, shuffle_files=True)
        self.val = tfds.load(dataset_name, split='train' if dataset_name=="iris" else "test", as_supervised=True, shuffle_files=True)
        self.classes = 3 if dataset_name == "iris" else 12
        self.signal_len = 4 if dataset_name == "iris" else min(max_sequence_length, 16000)
        self.train = self.train.map(self.reshape).take(1024).padded_batch(batch_size, padded_shapes=([self.signal_len, 1], ())).prefetch(tf.data.AUTOTUNE)
        self.val = self.val.map(self.reshape).take(1024).padded_batch(1, padded_shapes=([self.signal_len, 1], ())).prefetch(tf.data.AUTOTUNE)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = []

    def reshape(self, x, y):
        x = tf.dtypes.cast(tf.reshape(x, [-1, 1])[:self.signal_len], tf.float32)
        return x, y

if __name__ == "__main__":
    print(ClassificationDataset("speech_commands").classes)
    for x, y in ClassificationDataset("speech_commands").train:
        print(x.shape, y.shape)

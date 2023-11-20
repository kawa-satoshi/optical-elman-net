import numpy as np
import tensorflow as tf


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        df,
        val_size=0.33,
        label_columns=None,
        batch_size=32,
        normalize=True,
        sigma_magnitude=1.0,
    ):
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = [tf.keras.metrics.MeanAbsoluteError("l1")]
        
        # Store the raw data.
        n = len(df)
        self.train_df = df[0 : int(n * (1 - val_size))]
        self.val_df = df[int(n * (1 - val_size)) :]
        self.batch_size = batch_size

        if normalize:
            train_mean = self.train_df.mean()
            train_std = self.train_df.std()
            self.train_df = (self.train_df - train_mean) / train_std * sigma_magnitude
            self.val_df = (self.val_df - train_mean) / train_std * sigma_magnitude

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, batch_size):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=batch_size,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, batch_size=self.batch_size)

    @property
    def val(self):
        return self.make_dataset(self.val_df, batch_size=1)

import tensorflow as tf

class MemoryDataset:
    def __init__(
        self,
        n: int,
        k: int,
        batch_size = 32,
        num_samples: int = 1000,
    ):
        def data_generator():
            for _ in range(num_samples):
                x = tf.random.uniform(shape=(n, 1), minval=-1, maxval=1, dtype=tf.float32)
                y = x[n-k:n-k+1, :]
                yield x, y
        
        self.train = tf.data.Dataset.from_generator(data_generator, output_shapes=((n, 1), (1, 1)), output_types=(tf.float32, tf.float32)).batch(batch_size)
        self.test = self.val = tf.data.Dataset.from_generator(data_generator, output_shapes=((n, 1), (1, 1)), output_types=(tf.float32, tf.float32)).batch(1)
        
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = []

if __name__ == "__main__":
    self = MemoryDataset(4, 1, 1)
    for x, y in self.train.take(1):
        print("--", x, y)

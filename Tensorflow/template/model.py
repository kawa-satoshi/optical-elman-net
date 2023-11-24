from typing import Literal
import tensorflow as tf

def custom_sqrt(x, max_value=255.0):
    x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=max_value)
    return tf.sqrt(x / max_value) * max_value

def get_model(hidden_size: int, output_size: int, input_shape: list[int], activation: Literal["relu"] | Literal["tanh"] | Literal["custom"] = "relu", custom_max_value=255.0):
    if activation == "custom":
        activation = "linear"
        custom_function = lambda x: custom_sqrt(x, custom_max_value)
    else: custom_function = lambda x: x
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape),
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False, activation=activation),
        tf.keras.layers.Lambda(custom_function),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=output_size)
    ])
    return model

def get_classification_model(hidden_size: int, output_size: int, input_shape: list[int], activation: Literal["relu"] | Literal["tanh"] | Literal["custom"] = "relu", custom_max_value=255.0):
    model = get_model(hidden_size, output_size, input_shape, activation, custom_max_value)
    model.add(tf.keras.layers.Softmax())
    return model

if __name__ == "__main__":
    model = get_classification_model(10, 3, [1, 4])
    print(model.summary())

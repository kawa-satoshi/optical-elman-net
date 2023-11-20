# Helper function to run inference on a TFLite model
from typing import List, Tuple
import numpy as np
from template.dataset.window_generator import WindowGenerator
from template.dataset.classification_dataset import ClassificationDataset
import tensorflow as tf

def calc_MC(results: List[Tuple[np.ndarray, np.ndarray]]) -> np.floating:
    x_mean = np.mean(np.array([x for x,y in results]))
    y_mean = np.mean(np.array([y for x,y in results]))
    covariance = np.mean(np.array([(x - x_mean)*(y-y_mean) for x,y in results]))
    variance_x = np.mean(np.array([np.power(x - x_mean, 2) for x,y in results]))
    variance_y = np.mean(np.array([np.power(y - y_mean, 2) for x,y in results]))
    result = covariance / (variance_x * variance_y)
    return result

def run_tflite_model(tflite_file: str, dataset: WindowGenerator | ClassificationDataset, quantized:bool = False):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    results = []
    # Load a small subset of your training data
    for x, y in dataset.val:
        if type(dataset) == ClassificationDataset:
            y = tf.one_hot(y, depth=dataset.classes)
        # Check if the input type is quantized, then rescale input data to uint8
        if quantized:
            input_scale, input_zero_point = input_details["quantization"]
            x = x / input_scale + input_zero_point

        x = np.array(x).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if quantized:
            output_scale, output_zero_point = output_details["quantization"]
            output = output * output_scale - output_zero_point

        results.append((output[0], tf.reshape(y, [-1]).numpy()))
    l1 = np.mean(np.array([np.abs(x-y) for x,y in results]))
    l2 = np.mean(np.array([np.power(x-y, 2) for x,y in results]))
    memory = calc_MC(results)
    
    return l1, l2, memory

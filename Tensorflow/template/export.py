import pathlib
import tensorflow as tf
from template.dataset.window_generator import WindowGenerator

def representative_dataset_generator(dataset: WindowGenerator):
    def inner():
        # Load a small subset of your training data
        for sample_input, sample_label in dataset.val.take(100):
            # sample_input = # Load or generate a single input sample
            yield [sample_input]
    return inner

def save_model(model: tf.keras.models.Sequential, save_name: str, quantize: bool = False, dataset: WindowGenerator | None = None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_lower_tensor_list_ops = False
    
    if quantize:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter._experimental_lower_tensor_list_ops = False
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        assert dataset, "dataset should be set for quantization"
        converter.representative_dataset = representative_dataset_generator(dataset)
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    if quantize:
        tflite_model_file = tflite_models_dir/f"{save_name}_int8.tflite"
    else:
        tflite_model_file = tflite_models_dir/f"{save_name}.tflite"
    tflite_model_file.write_bytes(tflite_model)

    return tflite_model_file

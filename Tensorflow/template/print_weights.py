import tensorflow as tf

def print_weights(model: tf.keras.models.Sequential):
    def get_as_array(id: str):
        return model.get_weight_paths()[id].numpy().tolist()
    print("input weights = ".upper(), get_as_array("simple_rnn.cell.kernel"))
    print("\nrecurrent bias = ".upper(), get_as_array("simple_rnn.cell.bias"))
    print("recurrent weights = ".upper(), get_as_array("simple_rnn.cell.recurrent_kernel"))
    print("\noutput bias = ".upper(), get_as_array("dense.bias"))
    print("output weights = ".upper(), get_as_array("dense.kernel"))

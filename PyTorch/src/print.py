def print_weights(model, quantized=True):
    bias_ih = model.rnn.bias_ih
    bias_hh = model.rnn.bias_hh
    if quantized:
        weight_ih = model.rnn.get_weight()["weight_ih"].int_repr()
        weight_hh = model.rnn.get_weight()["weight_hh"].int_repr()
        weight_ho = model.linear.weight().int_repr()
        bias_ho = model.linear.bias()
    else:
        weight_ih = model.rnn.weight_ih
        weight_hh = model.rnn.weight_hh
        weight_ho = model.linear.weight
        bias_ho = model.linear.bias
    print("Bias input - hidden", bias_ih.tolist())
    print("Weight input - hidden", weight_ih.tolist())
    print("Bias hidden - hidden", bias_hh.tolist())
    print("Weight hidden - hidden", weight_hh.tolist())
    print("Bias hidden - output", bias_ho.tolist())
    print("Weight hidden - output", weight_ho.tolist())

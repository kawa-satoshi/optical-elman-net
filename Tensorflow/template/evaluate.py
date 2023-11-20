from template.export import save_model
from template.inference_tflite import run_tflite_model

def evaluate(model, dataset, dataset_name, ACTIVATION):
    tflite_model_file = save_model(model, save_name=f"{dataset_name}-{ACTIVATION}", quantize=False)
    l1, l2, memory = run_tflite_model(tflite_model_file, dataset)
    
    tflite_model_file_quantized = save_model(model, save_name=f"{dataset_name}-{ACTIVATION}", quantize=True, dataset=dataset)
    l1_quantized, l2_quantized, memory_quantized = run_tflite_model(tflite_model_file_quantized, dataset, quantized=True)

    is_memory = dataset_name == "memory"

    if is_memory:
        print(f"float32 model, L1 = {l1}, L2 = {l2}, Memory = {memory}")
        print(f"int8 model, L1 = {l1_quantized}, L2 = {l2_quantized}, Memory = {memory_quantized}")
    else:
        print(f"float32 model, L1 = {l1}, L2 = {l2}")
        print(f"int8 model, L1 = {l1_quantized}, L2 = {l2_quantized}")

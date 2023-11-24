from typing import Literal
import torch
from functools import partial
from src import print_weights, RNN, train, evaluate, SpeechCommands, Iris
from torch.utils.data import DataLoader
import sys

quantize = partial(torch.quantization.quantize_dynamic, qconfig_spec={torch.nn.RNNCell, torch.nn.Linear}, dtype=torch.qint8)

DATASET: Literal["speech_commands"] | Literal["iris"] = sys.argv[1]
ACTIVATION_FUNCTION: Literal["relu"] | Literal["custom"] | Literal["tanh"] = sys.argv[2]
HIDDEN_SIZE = int(sys.argv[3])
SEQUENCE_LENGTH = int(sys.argv[4])
EPOCHS = 100
SIGMA = 1.0
BATCH_SIZE = 1

MAX_SEQUENCE_LENGTH = 1600

if DATASET == "speech_commands":
    train_dataset = SpeechCommands(MAX_SEQUENCE_LENGTH, is_train=True)
    val_dataset = SpeechCommands(MAX_SEQUENCE_LENGTH, is_train=False)
else:
    train_dataset = Iris(is_train=True)
    val_dataset = Iris(is_train=False)

model = RNN(1, HIDDEN_SIZE, len(train_dataset.words), ACTIVATION_FUNCTION, sigma_magnitude=SIGMA, classification=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

quantized_model = quantize(model)
print("====== FLOAT32 MODEL RESULT 0 EPOCHS ======")
evaluate(model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("====== INT8 MODEL RESULT 0 EPOCHS ======")
evaluate(quantized_model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("--------------------------------------------")

train(model, train_dataloader, val_dataloader, 1, is_classification=True)
quantized_model = quantize(model)
print("====== FLOAT32 MODEL RESULT 1 EPOCHS ======")
evaluate(model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("====== INT8 MODEL RESULT 1 EPOCHS ======")
evaluate(quantized_model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("--------------------------------------------")

train(model, train_dataloader, val_dataloader, EPOCHS-1, is_classification=True)
quantized_model = quantize(model)
print("====== FLOAT32 MODEL RESULT 100 EPOCHS ======")
evaluate(model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("====== INT8 MODEL RESULT 100 EPOCHS ======")
evaluate(quantized_model, val_dataloader, is_classification=True, data=DATASET, af=ACTIVATION_FUNCTION, n_hid=HIDDEN_SIZE, n_seq=SEQUENCE_LENGTH)
print("--------------------------------------------")

print("====== FLOAT32 MODEL WEIGHTS ======")
print_weights(model, False)
print("====== INT8 MODEL WEIGHTS ======")
print_weights(quantized_model, True)

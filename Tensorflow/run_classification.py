import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from template.train import prepare_model
from template.print_weights import print_weights
from template.evaluate import evaluate

#DATASET_NAME = "speech_commands"
DATASET_NAME = "iris"

EPOCHS = 100
ACTIVATION = "relu"
HIDDEN_SIZE = 10
MAX_SEQUENCE_LENGTH = 1600

dataset, model = prepare_model(
    dataset_name=DATASET_NAME,
    activation=ACTIVATION,
    hidden_size=HIDDEN_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH
)

print("="*30, "EPOCH 0 result", "="*30)
evaluate(model, dataset, DATASET_NAME, ACTIVATION)

model.fit(dataset.train, epochs=1, validation_data=dataset.val)
print("="*30, "EPOCH 1 result", "="*30)
evaluate(model, dataset, DATASET_NAME, ACTIVATION)

model.fit(dataset.train, epochs=EPOCHS-1, validation_data=dataset.val)
print("="*30, f"EPOCH {EPOCHS} result", "="*30)
evaluate(model, dataset, DATASET_NAME, ACTIVATION)

print("="*30, "WEIGHTS START", "="*30)
print_weights(model)
print("="*30, "WEIGHTS END", "="*30)

evaluate(model, dataset, DATASET_NAME, ACTIVATION)

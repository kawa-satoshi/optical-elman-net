import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from template.train import prepare_model
from template.print_weights import print_weights
from template.evaluate import evaluate

DATASET_NAME = "SP500"
# DATASET_NAME = "airline-passengers"

EPOCHS = 100
ACTIVATION = "custom"
custom_max_value = 255.0
DATA_SIGMA = 255.0
HIDDEN_SIZE = 10
SEQUENCE_LENGTH = 20

dataset, model = prepare_model(
    dataset_name=DATASET_NAME,
    activation=ACTIVATION,
    hidden_size=HIDDEN_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    custom_max_value=custom_max_value,
    sigma_magnitude=DATA_SIGMA
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

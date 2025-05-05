import h5py
from tensorflow.keras.models import load_model

from utils import ValidationDataGenerator


model_paths = ['best_model301_enhanced_SE.h5', 'best_model301_enanced_complex.h5', 'best_model201_attention.h5', "best_model201_complex.h5", "best_model201_enanced_complex.h5"]

# 301-window models

chromosome_names = [i for i in range(1,23)] + ["X", "Y"]

val_generator = ValidationDataGenerator(
    [f'data/window_301/train_test_data{n}.h5' for n in chromosome_names]
)

model = load_model(model_paths[0])
loss, accuracy = model.evaluate(val_generator, verbose=1, steps=30)
print(f'{model_paths[0]}Validation accuracy: {accuracy:.4f}')


model = load_model(model_paths[1])
loss, accuracy = model.evaluate(val_generator, verbose=1, steps=30)
print(f'{model_paths[1]}Validation accuracy: {accuracy:.4f}')


# 201-window models
val_generator = ValidationDataGenerator(
    [f'data/window_201/train_test_data{n}.h5' for n in chromosome_names]
)

model = load_model(model_paths[2])
loss, accuracy, _, _ = model.evaluate(val_generator, verbose=1, steps=30)
print(f'{model_paths[2]}Validation accuracy: {accuracy:.4f}')

model = load_model(model_paths[3])
loss, accuracy, _, _ = model.evaluate(val_generator, verbose=1, steps=30)
print(f'{model_paths[3]}Validation accuracy: {accuracy:.4f}')

model = load_model(model_paths[4])
loss, accuracy = model.evaluate(val_generator, verbose=1, steps=30)
print(f'{model_paths[4]}Validation accuracy: {accuracy:.4f}')





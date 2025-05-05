from tensorflow.keras.models import load_model

from utils import ValidationDataGenerator


chromosome_names = [i for i in range(1, 8)]

val_generator = ValidationDataGenerator(
    [f"MusMusculus/window_201/train_test_data{n}.h5" for n in chromosome_names]
)

model = load_model("best_model201_enanced_complex.h5")
loss, accuracy = model.evaluate(val_generator, verbose=1, steps=30)
print(f"Validation accuracy: {accuracy:.4f}")

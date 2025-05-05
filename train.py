import pickle

import numpy as np
import tensorflow as tf

from models import (build_attention_model, build_complex_model, enhanced_301,
                    enhanced_cnn)
from utils import ChromosomeDataGenerator, ValidationDataGenerator

WINDOW_SIZE = 301  # 201
chromosome_names = [i for i in range(1, 23)] + ["X", "Y"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


val_generator = ValidationDataGenerator(
    [f"data/window_{WINDOW_SIZE}/train_test_data{n}.h5" for n in chromosome_names]
)

all_chr_files = [
    f"data/window_{WINDOW_SIZE}/train_test_data{n}.h5" for n in chromosome_names
]
# model = enhanced_301((WINDOW_SIZE, 4))
model = enhanced_cnn((WINDOW_SIZE, 4))
# model = build_complex_model((WINDOW_SIZE, 4))
# model = build_attention_model((WINDOW_SIZE, 4))

history = model.fit(
    ChromosomeDataGenerator(all_chr_files),
    epochs=100,
    steps_per_epoch=1000,
    validation_data=val_generator,
    validation_steps=20,  # len(val_generator), # 20 should be enough
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"best_model{WINDOW_SIZE}_enhanced_SE.h5", save_best_only=True
        ),
    ],
)


with open(f"history_{WINDOW_SIZE}_enahnced_SE.pkl", "wb") as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

import pandas as pd
import numpy as np
from multiprocessing import Queue
from pathlib import Path

import tensorflow as tf
from keras_cv_attention_models import beit
from keras.layers import Dense
from keras.regularizers import L2
from keras.models import Model
from keras.optimizers.optimizer_experimental.adamw import AdamW


WEIGHTS_PATH = Path(__file__).parents[1] / 'weights' / 'beitv2_weights.h5'
AUTOTUNE = tf.data.experimental.AUTOTUNE

ordered_genera = ['Aneurhynchus', 'Basalys', 'Coptera', 'Entomacis', 'Idiotypa', 'Ismarus',
                  'Monelata', 'Other-Hymenoptera', 'Paramesius', 'Psilus', 'Spilomicrus', 'Trichopria']
ordered_sexes = ['female', 'male', 'unknown']

def get_model():
    beit_model = beit.BeitV2LargePatch16(
        input_shape=(224, 224, 3),
        pretrained=None,
        num_classes=0,
        drop_connect_rate=0.3,
    )

    regularization = L2(l2=0.0005)
    n_genera = len(ordered_genera)
    n_sexes = len(ordered_sexes)

    x_genus = Dense(units=n_genera, activation='softmax', kernel_regularizer=regularization,
                    name="genus")(beit_model.output)
    x_sex = Dense(units=n_sexes, activation='softmax', kernel_regularizer=regularization,
                  name="sex")(beit_model.output)

    model = Model(inputs=beit_model.input, outputs=[x_genus, x_sex])

    model.compile(
        optimizer=AdamW(learning_rate=0.0001, beta_1=0.95),
        loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
        metrics={'output1': 'accuracy', 'output2': 'accuracy'},
    )

    return model, beit_model.preprocess_input

def decode_predictions(genus_preds: np.ndarray, sex_preds: np.ndarray):
    genus_decoder = dict(enumerate(ordered_genera))
    sex_decoder = dict(enumerate(ordered_sexes))
    genus_idx = np.argmax(genus_preds, axis=1)
    sex_idx = np.argmax(sex_preds, axis=1)

    genera = [genus_decoder[idx] for idx in genus_idx]
    sexes = [sex_decoder[idx] for idx in sex_idx]

    return genera, sexes

def predict(imgs: np.ndarray, queue: Queue):
    model, preprocess_func = get_model()
    model.load_weights(WEIGHTS_PATH)
    ds = tf.data.Dataset.from_tensor_slices(preprocess_func(imgs)).batch(4).prefetch(buffer_size=AUTOTUNE)
    genus_preds, sex_preds = model.predict(ds)

    genera, sexes = decode_predictions(genus_preds, sex_preds)
    genus_idx = np.argmax(genus_preds, axis=1)
    sex_idx = np.argmax(sex_preds, axis=1)
    genus_scores = [genus_preds[i, idx] for i, idx in enumerate(genus_idx)]
    sex_scores = [sex_preds[i, idx] for i, idx in enumerate(sex_idx)]

    results = pd.DataFrame({
        'Genus': genera,
        'Genus Score': genus_scores,
        'Sex': sexes,
        'Sex Score': sex_scores
    })

    queue.put(results)

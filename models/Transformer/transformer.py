from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


class Transformer_ENC:

    def __init__(self, head_size: int, num_heads: int, ff_dim, dropout: float = 0.):
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def encoder(self, input):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(input)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + input

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=input.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self,
                    num_transformer_blocks: int,
                    mlp_units: list,
                    mlp_dropout: float = 0.):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)


input_shape = x_train.shape[1:]


model = Transformer_ENC(head_size=256,
                        num_heads=4,
                        ff_dim=4,
                        dropout=0.25)

model.encoder = model.build_model(
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4
)

model.encoder.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)

cwd = Path().absolute()
checkpoint_filepath = f'{cwd}/exp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.encoder.load_weights(checkpoint_filepath)

model.encoder.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), model_checkpoint_callback]

x_test, y_test = None, None
x_train, y_train = None, None

model.encoder.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.encoder.evaluate(x_test, y_test, verbose=1)


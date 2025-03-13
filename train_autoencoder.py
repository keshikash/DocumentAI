# train_autoencoder.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import optuna

# Load cropped images
cropped_images = np.load("cropped_images.npy")

# Optuna objective function
def objective(trial):
    kernel_size = trial.suggest_categorical("kernel_size", [(3, 3), (5, 5), (7, 7)])
    num_epochs = trial.suggest_int("epochs", 20, 50)

    input_shape = cropped_images.shape[1:]
    encoder_input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size, activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPool2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(32, kernel_size, activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D()(x)
    decoded = Conv2D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(encoder_input, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    history = autoencoder.fit(
        cropped_images, cropped_images, epochs=num_epochs,
        batch_size=128, validation_split=0.25, verbose=1
    )

    val_loss = history.history['val_loss'][-1]
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return val_loss

# Hyperparameter tuning with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

# Get best hyperparameters and train final model
best_kernel_size = study.best_trial.params["kernel_size"]
best_epochs = study.best_trial.params["epochs"]

input_shape = cropped_images.shape[1:]
encoder_input = Input(shape=input_shape)
x = Conv2D(32, best_kernel_size, activation='relu', padding='same')(encoder_input)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(32, best_kernel_size, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPool2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(32, best_kernel_size, activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
x = Conv2D(32, best_kernel_size, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D()(x)
decoded = Conv2D(1, best_kernel_size, activation='sigmoid', padding='same')(x)

autoencoder = Model(encoder_input, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint("denoising_model_optuna.keras", save_best_only=True)
history = autoencoder.fit(
    cropped_images, cropped_images, epochs=best_epochs,
    batch_size=128, validation_split=0.25, callbacks=[checkpoint], verbose=2
)

autoencoder.save('denoising_model_optuna_final.keras')
print("Final model saved as 'denoising_model_optuna_final.keras'")

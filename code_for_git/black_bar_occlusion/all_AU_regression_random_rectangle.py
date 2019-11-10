import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
import h5py
from keras.models import Model
from helper import Metrics, two_rectangles_occluder

# load data
x_train = np.load("E:\\data\\np_data\\x_train_without_occlusion.npy")
y_train = np.load("E:\\data\\np_data\\y_train_without_occlusion.npy")
x_val = np.load("E:\\data\\np_data\\x_val_without_occlusion.npy")
y_val = np.load("E:\\data\\np_data\\y_val_without_occlusion.npy")

# data normalization
x_train_n3 = (x_train - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)
x_val_n3 = (x_val - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)

metrics = Metrics()

# pass randon_rectangles_occluder to datagenerator preposessor
datagen = ImageDataGenerator(preprocessing_function=two_rectangles_occluder)

datagen.fit(x_train_n3)

input_main = Input(shape=(128, 128, 3))
x = Conv2D(32, kernel_size=(3, 3), activation='relu') (input_main)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
out = Dense(26)(x)
out_last = Activation('linear')(out)
model = Model(inputs=input_main, outputs=out_last)

#tensorboard callback
tensorboard = TensorBoard(log_dir="E:\\thesis_work\\logs_02\\bosphorus_logs\\without_occlusion")

model.compile(optimizer='adam',
              loss='mse', metrics=['mse'])

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train_n3, y_train, batch_size=100),
                    steps_per_epoch=len(x_train_n3) / 100, epochs=30, verbose=2,
                    validation_data=(x_val_n3, y_val), callbacks=[metrics])

model.save("E:\\thesis_work\\models_02\\bosphorus_models\\without_occlusion.h5")

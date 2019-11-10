import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Activation, Reshape
from keras.callbacks import TensorBoard
import h5py
from keras.utils import to_categorical
from keras.models import Model
from helper import Metrics_2, bosphorus_face_parts_occluder


# load data
x_train = np.load("E:\\thesis_work\\bosphorus\\aligned_npy\\x_train_aligned.npy")
y_train = np.load("E:\\thesis_work\\bosphorus\\aligned_npy\\y_train_aligned.npy")
x_val = np.load("E:\\thesis_work\\bosphorus\\aligned_npy\\x_val_aligned.npy")
y_val = np.load("E:\\thesis_work\\bosphorus\\aligned_npy\\y_val_aligned.npy")

# data normalization
x_train_n3 = (x_train - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)
x_val_n3 = (x_val - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)

y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)

# train and val Data generator
datagen_train = ImageDataGenerator(preprocessing_function=bosphorus_face_parts_occluder)
datagen_val = ImageDataGenerator()

train_datagenerator = datagen_train.flow(x_train_n3, y_train_onehot, batch_size=100)
val_datagenerator = datagen_val.flow(x_val_n3, y_val_onehot, batch_size=100)

metrics = Metrics_2(val_data=val_datagenerator)

input_main = Input(shape=(100, 100, 3))
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
xc = Dense(128, activation='relu')(x)
xc = Dropout(0.3)(xc)
xc = Dense(256, activation='relu')(xc)
xc = Dropout(0.3)(xc)
out = Dense(26*6)(x)
out = Reshape((26, 6), input_shape=(26*6,))(out)
out_last = Activation('softmax')(out)
model = Model(inputs=input_main, outputs=out_last)

#tensorboard callback
tensorboard = TensorBoard(log_dir="E:\\thesis_work\\logs_02\\bosphorus_face_parts_occluded\\without_occlusion")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_datagenerator,
                    steps_per_epoch=len(x_train_n3) / 100, epochs=30, verbose=2,
                    validation_data=val_datagenerator, validation_steps=len(x_val_n3) / 100,
                    callbacks=[metrics, tensorboard])

model.save("E:\\thesis_work\\models_02\\bosphorus_face_parts_occluded\\without_occlusion.h5")

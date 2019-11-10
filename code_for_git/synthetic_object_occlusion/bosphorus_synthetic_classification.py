from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
import h5py
from keras.utils import to_categorical
from keras.models import Model
from helper import Metrics_classification
from augmentation import load_occluders, occlude_with_objects

# load data
x_train = np.load("E:\\thesis_work\\bosphorus\\np_data\\x_train.npy")
y_train = np.load("E:\\thesis_work\\bosphorus\\np_data\\y_train.npy")
x_val = np.load("E:\\thesis_work\\bosphorus\\np_data\\x_val.npy")
y_val = np.load("E:\\thesis_work\\bosphorus\\np_data\\y_val.npy")

y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)

# data normalization
x_train_n3 = (x_train - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)
x_val_n3 = (x_val - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)

metrics = Metrics_classification()

file_path = 'VOC2011'
occluders = load_occluders(pascal_voc_root_path=file_path)
def random_synthetic_occlusion(image):
    """"
    take image and apply random synthetic occlusion
    return occluded image
    """
    occluded_image = occlude_with_objects(image, occluders)
    return occluded_image

# pass randon_rectangles_occluder to datagenerator preposessor
datagen = ImageDataGenerator(preprocessing_function=random_synthetic_occlusion)
datagen.fit(x_train_n3)

# model
input_main = Input(shape=(100, 100, 3))
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_main)
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
out = Dense(26*6)(x)
out = Reshape((26, 6), input_shape=(26*6,))(out)
out_last = Activation('softmax')(out)
model = Model(inputs=input_main, outputs=out_last)

#tensorboard callback
tensorboard = TensorBoard(log_dir="E:\\thesis_work\\synthetic_occlusion\\bosphorus_logs\\three_60")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train_n3, y_train_onehot, batch_size=100),
                    steps_per_epoch=len(x_train_n3) / 100, epochs=30, verbose=2,
                    validation_data=(x_val_n3, y_val_onehot), callbacks=[metrics, tensorboard])

model.save("E:\\thesis_work\\synthetic_occlusion\\bosphorus_models\\three_60.h5")
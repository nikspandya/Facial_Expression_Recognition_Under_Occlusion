import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, concatenate
import h5py
from keras.utils import to_categorical
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from helper import random_rectangles_occluder, get_segmentation_mask

# load data
image_train = np.load("E:\\thesis_work\\RAF-DB\\np_data\\image_train.npy")
y_train = np.load("E:\\thesis_work\\RAF-DB\\np_data\\y_train_RAF-DB.npy")
image_val = np.load("E:\\thesis_work\\RAF-DB\\np_data\\image_val.npy")
y_val = np.load("E:\\thesis_work\\RAF-DB\\np_data\\y_test_RAF-DB.npy")
mask_train = np.load("E:\\thesis_work\\RAF-DB\\np_data\\mask_train.npy")
mask_val = np.load("E:\\thesis_work\\RAF-DB\\np_data\\mask_val.npy")
#covert label 1-7 to 0-6
y_train = y_train - 1
y_val = y_val -1

#onehot encoding
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)

#initialize three data-generator
image_gen = ImageDataGenerator(preprocessing_function=random_rectangles_occluder)
mask_gen = ImageDataGenerator(preprocessing_function=get_segmentation_mask)
image_gen_empty = ImageDataGenerator()

# get parallel augmantation same for both occlusion and segmentation label
def generate_data_generator(genX, genY1, X, Y1, Y2):
    genX = genX.flow(X, batch_size=100, seed=2)
    genY1 = genY1.flow(Y1, Y2,batch_size=100, seed=2)
    while True:
            Xi = genX.next()
            Yi1 = genY1.next()
            yield Xi, [Yi1[1], Yi1[0]]

train_gen = generate_data_generator(image_gen, mask_gen, image_train, mask_train, y_train_onehot)
valid_gen = generate_data_generator(image_gen_empty, image_gen_empty, image_val, mask_val, y_val_onehot)

# model without skip connection to flow segmentation information
inputs = Input((96, 96, 3))
c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2, 2))(c1)
c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
p2 = MaxPooling2D((2, 2))(c2)
c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
p3 = MaxPooling2D((2, 2))(c3)
c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
p5 = MaxPooling2D(pool_size=(2, 2))(c5)
c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
c7 = Conv2D(128, (3, 3), activation='relu',  padding='same')(u7)
u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
c8 = Conv2D(64, (3, 3), activation='relu',  padding='same')(u8)
u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
c9 = Conv2D(32, (3, 3), activation='relu',  padding='same')(u9)
output_1 = Conv2D(1, (1, 1), activation='sigmoid',  name="out_seg")(c9)
x1 = Flatten()(p5)
x2 = Flatten()(c9)
x = concatenate([x1, x2])
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
# concatenate features before classification output
output_2 = Dense(6,activation='softmax',  name="out_classi")(x)
model = Model(inputs=inputs, outputs=[output_2, output_1])

tensorboard = TensorBoard(log_dir="E:\\thesis_work\\multi-task_learning\\raf_db\\logs\\two_30_new_combine")

model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_gen,
                    steps_per_epoch=len(image_train)//100,
                    epochs=70,
                    validation_data=valid_gen,
                    validation_steps=len(image_val)//100, verbose=2, callbacks=[tensorboard])

model.save("E:\\thesis_work\\multi-task_learning\\raf_db\\models\\two_30_new_combine.h5")
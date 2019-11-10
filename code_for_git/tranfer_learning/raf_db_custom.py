import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
import h5py
from keras.utils import to_categorical
from keras.models import Model

# load data
x_train = np.load("E:\\thesis_work\\RAF-DB\\np_data\\x_train_RAF-DB.npy")
y_train = np.load("E:\\thesis_work\\RAF-DB\\np_data\\y_train_RAF-DB.npy")
x_val = np.load("E:\\thesis_work\\RAF-DB\\np_data\\x_test_RAF-DB.npy")
y_val = np.load("E:\\thesis_work\\RAF-DB\\np_data\\y_test_RAF-DB.npy")

#covert label 1-7 to 0-6
y_train = y_train - 1
y_val = y_val -1

#onehot encoding
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)

# data normalization
x_train_n3 = (x_train - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)
x_val_n3 = (x_val - x_train.mean(axis=(0,1,2), keepdims=True)) / x_train.std(axis=(0,1,2), keepdims=True)

datagen = ImageDataGenerator()
datagen.fit(x_train_n3)

# custom model same as VGG16
input_main = Input(shape=(100, 100, 3))
x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_main)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
out = Dense(6)(x)
out_last = Activation('softmax')(out)
model = Model(inputs=input_main, outputs=out_last)

# freeze same layers as in pre-trained model
for layer in model.layers[10:18]:
    layer.trainable = False

#tensorboard callback
tensorboard = TensorBoard(log_dir="E:\\thesis_results\\tranfer_learning\\raf_db\\logs\\custom_model_small")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train_n3, y_train_onehot, batch_size=100),
                    steps_per_epoch=len(x_train_n3) / 100, epochs=20, verbose=2,
                    validation_data=(x_val_n3, y_val_onehot), callbacks=[tensorboard])

model.save("E:\\thesis_results\\tranfer_learning\\raf_db\\models\\custom_model_small.h5")
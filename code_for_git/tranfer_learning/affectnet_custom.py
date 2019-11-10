from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, Softmax
from keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate
from keras.callbacks import TensorBoard
import h5py
from keras.models import Model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# data-generator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\train",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=100,
    class_mode="categorical",
    shuffle=True,
    seed=42)
valid_generator = val_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\val",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=100,
    class_mode="categorical",
    shuffle=False,
    seed=42)

# Custom VGG16 with random weights
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

tensorboard = TensorBoard(log_dir="E:\\thesis_results\\tranfer_learning\\affectnet\\logs\\custom_model")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10, verbose=2, callbacks=[tensorboard])

model.save("E:\\thesis_results\\tranfer_learning\\affectnet\\models\\custom_model.h5")
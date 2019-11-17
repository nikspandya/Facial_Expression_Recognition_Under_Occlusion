# Imports
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
import h5py
from keras.models import Model
from helper import two_rectangles_occluder, rectangle_occluder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# hyper parameters
batch_size_ = 512
num_epochs = 10
img_rows, img_cols = 128, 128

# Train & validation Generator
train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=two_rectangles_occluder)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\train",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=batch_size_,
    class_mode="categorical",
    shuffle=True,
    seed=42)
valid_generator = val_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\val",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=batch_size_,
    class_mode="categorical",
    shuffle=False,
    seed=42)

# Model
input_main = Input(shape=(img_rows, img_cols, 3))
x = Conv2D(16, kernel_size=(3, 3), activation='relu')(input_main)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
out = Dense(6)(x)
out_last = Activation('softmax')(out)
model = Model(inputs=input_main, outputs=out_last)

tensorboard = TensorBoard(log_dir="E:\\thesis_results\\fixed_val_occluded\\affectnet\\logs\\train_val_without")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=num_epochs, verbose=2, callbacks=[tensorboard])

model.save("E:\\thesis_results\\fixed_val_occluded\\affectnet\\models\\train_val_without.h5")

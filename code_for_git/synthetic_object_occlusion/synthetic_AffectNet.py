from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
import h5py
from keras.models import Model
from augmentation import load_occluders,occlude_with_objects
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

file_path = 'VOC2011'
occluders = load_occluders(pascal_voc_root_path=file_path)
def random_synthetic_occlusion(image):
    """"
    take image and apply random synthetic occlusion
    return occluded image
    """
    occluded_image = occlude_with_objects(image, occluders)
    return occluded_image

train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=random_synthetic_occlusion)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\train",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=500,
    class_mode="categorical",
    shuffle=True,
    seed=42)
valid_generator = val_datagen.flow_from_directory(
    directory="E:\\thesis_work\\AffectNet\\object_occluded_val\\two_40",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=500,
    class_mode="categorical",
    shuffle=False,
    seed=42)

input_main = Input(shape=(100, 100, 3))
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

#tensorboard callback
tensorboard = TensorBoard(log_dir="E:\\thesis_results\\fixed_val_occluded\\affectnet\\logs\\two_object_40")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10, verbose=2, callbacks=[tensorboard])

model.save("E:\\thesis_results\\fixed_val_occluded\\affectnet\\models\\two_object_40.h5")

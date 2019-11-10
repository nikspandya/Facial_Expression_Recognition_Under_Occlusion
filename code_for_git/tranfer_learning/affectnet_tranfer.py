from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
import h5py
from keras.models import Model
from keras.applications.vgg16 import VGG16
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

# VGG16 with pre-trained weights
base_model = VGG16(weights = "imagenet", include_top=False, input_shape = (100, 100, 3))
base_model.load_weights("C:\\Users\\student\\.keras\\models\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
x = base_model.output
x = Flatten()(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation="softmax")(x)
model = Model(inputs = base_model.input, outputs = predictions)

tensorboard = TensorBoard(log_dir="E:\\thesis_results\\tranfer_learning\\affectnet\\logs\\vgg_with_weights")

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10, verbose=2, callbacks=[tensorboard])

model.save("E:\\thesis_results\\tranfer_learning\\affectnet\\models\\vgg_with_weights.h5")
import os
import math
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

ROOT_DIR = './PCOS'

def number_of_images_per_class():
    number_of_images = {}
    for dir in os.listdir(ROOT_DIR):
        number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))
    return number_of_images

def datafolder(path, split, number_of_images):
    if not os.path.exists(path):
        os.makedirs(path)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs(os.path.join(path, dir))
            for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)),
                                        size=(math.floor(split * number_of_images[dir]) - 5), replace=False):
                src = os.path.join(ROOT_DIR, dir, img)
                dest = os.path.join(path, dir)
                shutil.copy(src, dest)
                os.remove(src)
    else:
        print("Folder already exists")

def preprocessing_image(path, augment=False):
    if augment:
        image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input,
                                        horizontal_flip=True)
    else:
        image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    return image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

def train_model(train_data, val_data):
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(units=1, activation='sigmoid')(x)

    model = Model(base_model.input, x)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    mc = ModelCheckpoint(filepath='image_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, verbose=1)
    cb = [mc, es]

    model.fit(train_data, steps_per_epoch=10, epochs=30, validation_data=val_data, validation_steps=16, callbacks=cb)
    return model

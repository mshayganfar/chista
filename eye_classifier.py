from keras import regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomCrop, RandomZoom, RandomContrast
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


class EyeClassifier:
    def __init__(self, batch_size, image_height, image_width) -> None:
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

    def temp_augment_images(self, augmented_folder, image_path, image_name, num_rows, num_columns):
        img = load_img(image_path+image_name)

        image_data = img_to_array(img)

        images_data = np.expand_dims(image_data, axis=0)

        datagen = ImageDataGenerator(width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     brightness_range=(0.6, 2.0),
                                     shear_range=10,
                                     zoom_range=0.2,
                                     rotation_range=40,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        train_generator = datagen.flow(images_data, batch_size=20)

        fig, axes = plt.subplots(num_rows, num_columns)

        for r in range(num_rows):
            for c in range(num_columns):
                image_batch = train_generator.next()

                image = image_batch[0].astype('uint8')

                axes[r, c].imshow(image)

                im = Image.fromarray(image)
                im.save(augmented_folder+f'{r}_{c}_augmented_'+image_name)

        fig.set_size_inches(10, 10)

        fig, axes = plt.subplots(num_rows, num_columns)

        for r in range(num_rows):
            for c in range(num_columns):
                img = load_img(augmented_folder +
                               f'{r}_{c}_augmented_'+image_name)

                axes[r, c].imshow(img)

        fig.set_size_inches(11, 10)
        plt.show()

    def prepare_image_dataset(self, eye_images, seed_value, validation_split):
        eye_train_images = tf.keras.preprocessing.image_dataset_from_directory(
            eye_images,
            labels='inferred',
            label_mode='int',
            class_names=['eye', 'not_eye'],
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.image_height, self.image_width),
            shuffle=True,
            seed=seed_value,
            validation_split=validation_split,
            subset="training"
        )

        eye_validation_images = tf.keras.preprocessing.image_dataset_from_directory(
            eye_images,
            labels='inferred',
            label_mode='int',
            class_names=['eye', 'not_eye'],
            color_mode='rgb',
            batch_size=batch_size,
            image_size=(self.image_height, self.image_width),
            shuffle=True,
            seed=seed_value,
            validation_split=validation_split,
            subset="validation"
        )

        return eye_train_images, eye_validation_images

    def create_eye_model(self):
        eye_model = Sequential()

        # CONVOLUTIONAL LAYER
        eye_model.add(Conv2D(filters=32,
                             kernel_size=(3, 3), input_shape=(50, 50, 3)))

        # BATCH NORMALIZATION
        eye_model.add(BatchNormalization())

        eye_model.add(Activation('relu'))

        # POOLING LAYER
        eye_model.add(MaxPool2D(pool_size=(2, 2)))

        # CONVOLUTIONAL LAYER
        eye_model.add(Conv2D(filters=64,
                             kernel_size=(3, 3)))

        # BATCH NORMALIZATION
        eye_model.add(BatchNormalization())

        eye_model.add(Activation('relu'))

        # POOLING LAYER & DROPOUT
        eye_model.add(MaxPool2D(pool_size=(2, 2)))

        # CONVOLUTIONAL LAYER
        eye_model.add(Conv2D(filters=128,
                             kernel_regularizer=regularizers.l1_l2(
                                 l1=0.001, l2=0.001),
                             kernel_size=(3, 3)))

        # BATCH NORMALIZATION
        eye_model.add(BatchNormalization())

        eye_model.add(Activation('relu'))

        # POOLING LAYER & DROPOUT
        eye_model.add(MaxPool2D(pool_size=(2, 2)))

        eye_model.add(Dropout(0.2))

        # FLATTEN IMAGES FROM 50 by 50 to 2500 BEFORE FINAL LAYER
        eye_model.add(Flatten(input_shape=(50, 50, 3)))

        # 128 NEURONS IN DENSE HIDDEN LAYER
        eye_model.add(Dense(128,
                            kernel_regularizer=regularizers.l1_l2(
                                l1=0.001, l2=0.001),
                            activation='relu'))

        eye_model.add(Dropout(0.2))

        # 64 NEURONS IN DENSE HIDDEN LAYER
        eye_model.add(Dense(64, activation='relu'))

        # LAST LAYER IS THE CLASSIFIER
        eye_model.add(Dense(1, activation='sigmoid'))

        return eye_model


if __name__ == "__main__":

    batch_size = 32
    image_height = 50
    image_width = 50

    eye_classifier = EyeClassifier(batch_size, image_height, image_width)

    augmented_folder = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/augmented_images/'
    eye_images = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/Classes/Eye/'
    image_path = eye_images+'eye/'
    image_name = 'small__themakeupdoll-1759665766548332840.jpg'
    num_rows = 5
    num_columns = 4

    eye_classifier.temp_augment_images(
        augmented_folder, image_path, image_name, num_rows, num_columns)

    train_images, validation_images = eye_classifier.prepare_image_dataset(
        eye_images, 42, 0.25)
    
    eye_model = eye_classifier.create_eye_model()

    print(eye_model.summary())

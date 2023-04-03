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

        eye_model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        return eye_model

    def train_model(self, eye_model, train_images, validation_images, num_epochs):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        eye_history = eye_model.fit(train_images,
                                    validation_data=validation_images,
                                    epochs=num_epochs,
                                    callbacks=[early_stop])
        return eye_history

    def visualize_accuracy_loss(self, eye_history):
        acc = eye_history.history['accuracy']
        val_acc = eye_history.history['val_accuracy']

        loss = eye_history.history['loss']
        val_loss = eye_history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def prepare_test_image_dataset(self, test_eye_images, seed_value):
        eye_test_images = tf.keras.preprocessing.image_dataset_from_directory(
            test_eye_images,
            labels='inferred',
            label_mode='int',
            class_names=['eye', 'not_eye'],
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=(self.image_height, self.image_width),
            shuffle=True,
            seed=seed_value,
            validation_split=None,
            subset=None
        )
        return eye_test_images

    @classmethod
    def test_classifier(cls, eye_model, eye_test_images):
        eye_predictions = eye_model.predict(eye_test_images)
        print(eye_predictions)
        return eye_predictions

    @classmethod
    def show_results(cls, eye_model, eye_test_images):
        plt.figure(figsize=(14, 12))
        for images, labels in eye_test_images.take(1):
            for i in range(len(images)):
                eye_prediction = eye_model.predict(
                    tf.reshape(images[i], [-1, 50, 50, 3]))
                ax = plt.subplot(6, 4, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                eye_label = 'not_eye' if eye_prediction[0][0] > 0.5 else 'eye'
                image_title = eye_test_images.class_names[labels[i]]+' >> '+eye_label+' ('+str(
                    round(eye_prediction[0][0], 2))+')'
                plt.title(image_title)
                plt.axis("off")
        plt.show()


if __name__ == "__main__":

    batch_size = 32
    image_height = 50
    image_width = 50

    eye_classifier = EyeClassifier(batch_size, image_height, image_width)

    augmented_folder = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/augmented_images/'
    eye_images = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/Classes/Eye/'
    test_eye_images = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Test/beauty/Eye/'
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

    eye_history = eye_classifier.train_model(
        eye_model, train_images, validation_images, 150)

    eye_classifier.visualize_accuracy_loss(eye_history)

    eye_test_images = eye_classifier.prepare_test_image_dataset(
        test_eye_images, 42)

    eye_classifier.test_classifier(eye_model, eye_test_images)

    eye_classifier.show_results(eye_model, eye_test_images)

    eye_model.save_weights('weights/eye_weights.h5')

    eye_model.save('models/eye_model.h5')

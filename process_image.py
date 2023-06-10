import cv2
from keras.models import load_model
import os
import tensorflow as tf


class Beauty:
    def __init__(self) -> None:
        self.eye_model = load_model('models/eye_model.h5')
        self.face_model = load_model('models/face_model.h5')
        self.hair_model = load_model('models/hair_model.h5')
        self.lips_model = load_model('models/lips_model.h5')
        self.nail_model = load_model('models/nail_model.h5')
        self.products_model = load_model('models/products_model.h5')
        self.dog_model = load_model('models/dog_model.h5')

    # Classify the input image category and return top three sub-categories and their probabilities
    def classify_image_category(self, image_path, image_filename):
        image_file = os.path.join(image_path, image_filename)

        image = cv2.imread(image_file)
        image = cv2.resize(image, (50, 50))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.expand_dims(image, 0)

        eye_prediction = self.eye_model.predict(image)
        face_prediction = self.face_model.predict(image)
        hair_prediction = self.hair_model.predict(image)
        lips_prediction = self.lips_model.predict(image)
        nail_prediction = self.nail_model.predict(image)
        product_prediction = self.products_model.predict(image)
        dog_prediction = self.dog_model.predict(image)

        models_results = []

        models_results.append((round(1.0 - eye_prediction[0][0], 3), 'eye'))
        models_results.append((round(1.0 - face_prediction[0][0], 3), 'face'))
        models_results.append((round(1.0 - hair_prediction[0][0], 3), 'hair'))
        models_results.append((round(1.0 - lips_prediction[0][0], 3), 'lips'))
        models_results.append((round(1.0 - nail_prediction[0][0], 3), 'nail'))
        models_results.append(
            (round(1.0 - product_prediction[0][0], 3), 'product'))
        models_results.append((round(1.0 - dog_prediction[0][0], 3), 'dog'))

        return models_results

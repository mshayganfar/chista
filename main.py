from eye_classifier import EyeClassifier
from keras.models import load_model

if __name__ == "__main__":

    batch_size = 32
    image_height = 50
    image_width = 50

    test_eye_images = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Test/beauty/Eye/'

    clf = EyeClassifier(batch_size, image_height, image_width)

    eye_model = clf.create_eye_model()

    saved_eye_model = load_model('models/eye_model.h5')

    eye_test_images = clf.prepare_test_image_dataset(test_eye_images, 42)

    EyeClassifier.test_classifier(saved_eye_model, eye_test_images)

    EyeClassifier.show_results(saved_eye_model, eye_test_images)

    print(saved_eye_model.summary())

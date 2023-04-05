from eye_classifier import EyeClassifier
import pytest


@pytest.fixture(scope="class")
def eye_model():
    batch_size = 32
    image_height = 50
    image_width = 50
    return EyeClassifier(batch_size, image_height, image_width)


class TestEyeClassifier:
    def test_create_eye_model(self, eye_model):
        assert eye_model.create_eye_model() != object()

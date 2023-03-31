import cv2
import os

class ImageManipulator:
    def __init__(self) -> None:
        pass

    # Image preparation
    @staticmethod
    def load_image(image_path, image_name):
        image = cv2.imread(os.path.join(image_path, image_name))
        return image
    
    @staticmethod
    def resize_image(src_image, width, height):
        dim = (width, height)
    
        # resize image
        resized_image = cv2.resize(src_image, dim, interpolation=cv2.INTER_AREA)
    
        return resized_image
    
    @staticmethod
    def save_image(image_path, image_name, image):
        cv2.imwrite(os.path.join(image_path , image_name), image)

    # Resize and save subsampled images into a new folder
    @staticmethod
    def save_resized_images(dst_image_folder, resized_image_folder):
    
        filenames = os.listdir(dst_image_folder + '.')
    
        for filename in filenames:
            if filename[-3:] == 'jpg':
                loaded_image  = ImageManipulator.load_image(dst_image_folder, filename)
                resized_image = ImageManipulator.resize_image(loaded_image, 50, 50)
                ImageManipulator.save_image(resized_image_folder, 'small_' + filename, resized_image)

if __name__ == "__main__":
    gen = ImageManipulator()
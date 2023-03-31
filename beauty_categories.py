import os
import random
import shutil


class BeautyCategories:

    def __init__(self) -> None:
        pass

    # Read image file names for each category
    def read_category_filenames(self, image_classes_folder_base):
        sampled_filenames_dict = {'eye': [], 'face': [],
                                  'hair': [], 'lips': [], 'nail': [], 'products': []}
        for category in list(sampled_filenames_dict.keys()):
            specific_folder = image_classes_folder_base + \
                category.capitalize() + '/' + category + '/'
            filenames = [filename for filename in os.listdir(
                specific_folder + '.') if filename.endswith('.jpg')]
            sampled_filenames_dict[category] = filenames

            print(f"Category {category} is done.")

        return sampled_filenames_dict

    # Populate files into other categories' folders
    def populate_other_categories(self, sampled_filenames_dict, image_classes_folder_base):
        for category in sampled_filenames_dict:
            number_of_files = len(sampled_filenames_dict[category])
            number_of_files_per_category = number_of_files//5
            for other_category in list(sampled_filenames_dict.keys()):
                if other_category != category:
                    number_of_files_in_other_category = len(
                        sampled_filenames_dict[other_category])
                    random_numbers = random.sample(
                        range(0, number_of_files_in_other_category-1), number_of_files_per_category)
                    for file_index in random_numbers:
                        filename = sampled_filenames_dict[other_category][file_index]
                        src_folder = image_classes_folder_base + other_category.capitalize() + '/' + \
                            other_category + '/'
                        dst_folder = image_classes_folder_base + category.capitalize() + '/not_' + \
                            category + '/'
                        if os.path.exists(src_folder + filename):
                            shutil.copy(src_folder + filename,
                                        dst_folder + filename)
                        else:
                            print(
                                f"File {filename} doesn't exist in this {src_folder}!")
            print(f"Category {category} is done.")


if __name__ == "__main__":

    beauty_category = BeautyCategories()

    image_classes_folder_base = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/Classes/'

    sampled_filenames_dict = beauty_category.read_category_filenames(
        image_classes_folder_base)

    # beauty_category.populate_other_categories(sampled_filenames_dict, image_classes_folder_base)

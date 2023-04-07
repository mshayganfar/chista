
import os
import pandas as pd
import random
import shutil

from image_manipulator import ImageManipulator


class Influencers:
    def __init__(self, seed_num) -> None:
        self.seed_number = seed_num

    # Randomly pick N usernames
    def get_random_usernames(self, df, influencers_count, num_of_influencers, verbose):
        username_list = []

        random.seed(self.seed_number)
        random_numbers = random.sample(
            range(0, influencers_count-1), num_of_influencers)

        for row_index in random_numbers:
            username_list.append(df.iloc[row_index].username)

        if verbose:
            print(username_list)

        return username_list

    # Extract image file names
    def extract_image_file_names(self, username_list, src_image_folder):

        filenames_list = []

        for username in username_list:
            first_char = username[0]
            if first_char.isalpha() == False:
                first_char = '_'
            # check if file exist in destination
            if os.path.exists(src_image_folder + first_char):
                specific_folder = src_image_folder + first_char + '/'
                filenames = [filename for filename in os.listdir(
                    specific_folder + '.') if filename.startswith(username)]
                filenames_list.append(filenames)
            else:
                print(f"Folder {first_char} doesn't exist!")

        return filenames_list

    # Subsample image files for all of the influencers
    def get_influencers_random_subsampled_files(self, filenames_list, num_of_files_per_influencer):

        subsampled_filename_list = []

        random.seed(self.seed_number)

        for i in range(0, len(filenames_list)):
            influencer_image_filenames = []
            random_numbers = random.sample(
                range(0, len(filenames_list[i])-1), num_of_files_per_influencer)

            for file_index in random_numbers:
                influencer_image_filenames.append(
                    filenames_list[i][file_index])

            subsampled_filename_list.append(influencer_image_filenames)

        return subsampled_filename_list

    # Copy the subsampled files into the destination folder
    @staticmethod
    def copy_influencers_files(subsampled_filenames, src_folder_base, dst_folder_base):

        for i in range(0, len(subsampled_filenames)):
            for j in range(0, len(subsampled_filenames[i])):
                filename = subsampled_filenames[i][j]
                first_char = filename[0]
                if first_char.isalpha() == False:
                    first_char = '_'
                # check if file exist in destination
                if os.path.exists(src_folder_base + first_char):
                    specific_src_folder = src_folder_base + first_char + '/'
                    shutil.copy(specific_src_folder + filename,
                                dst_folder_base + filename)
                else:
                    print(f"Folder {first_char} doesn't exist!")

    # Read N files of a specific influencer
    @staticmethod
    def get_influencer_files(image_folder, username):

        first_char = username[0]
        if first_char.isalpha() == False:
            first_char = '_'
        # check if file exist in destination
        if os.path.exists(image_folder + first_char):
            specific_folder = image_folder + first_char + '/'
            filenames = [filename for filename in os.listdir(
                specific_folder + '.') if filename.startswith(username)]
        else:
            print(f"Folder {first_char} doesn't exist!")

        return filenames

    # Subsample image files
    def get_influencr_random_subsampled_files(self, filenames, number_of_files):

        subsampled_filename_list = []

        random.seed(self.seed_number)

        try:
            random_numbers = random.sample(
                range(0, len(filenames)), number_of_files)
        except ValueError:
            print('WARNING: The requested number is bigger than the length of filenames!')
            random_numbers = random.sample(
                range(0, len(filenames)), min(len(filenames), number_of_files))

        for file_index in random_numbers:
            subsampled_filename_list.append(filenames[file_index])

        return subsampled_filename_list

    # Copy the subsampled files into the destination folder
    @staticmethod
    def copy_influencer_files(filenames, src_folder_base, dst_folder_base):

        for i in range(0, len(filenames)):
            filename = filenames[i]
            first_char = filename[0]
            if first_char.isalpha() == False:
                first_char = '_'
            # check if file exist in destination
            if os.path.exists(src_folder_base + first_char):
                specific_src_folder = src_folder_base + first_char + '/'
                shutil.copy(specific_src_folder + filename,
                            dst_folder_base + filename)
            else:
                print(f"Folder {first_char} doesn't exist!")


if __name__ == "__main__":

    num_of_influencers = 10
    num_of_files_per_influencer = 10

    influencers_data = '/Users/mshayganfar/sb_capstone/data/influencers.csv'

    src_image_folders_base = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/images/'
    dst_image_folders_base = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/subset_images/'
    resized_image_folders_base = '/Users/mshayganfar/Mahni_Folder/Mahni/Influencers/Beauty/resized_images/'

    df_influencers = pd.read_csv(influencers_data)

    beauty_influencers_count = df_influencers[df_influencers['category'] == 'beauty'].username.count(
    )

    print(f"Beauty influencers count: {beauty_influencers_count}")

    influencers = Influencers(42)
    usernames = influencers.get_random_usernames(
        df_influencers, beauty_influencers_count, num_of_influencers, True)

    filenames = influencers.extract_image_file_names(
        usernames, src_image_folders_base)

    influencers_subsampled_filenames = influencers.get_influencers_random_subsampled_files(
        filenames, num_of_files_per_influencer)

    # influencers.copy_influencers_files(
    #     influencers_subsampled_filenames, src_image_folders_base, dst_image_folders_base)

    # ImageManipulator.save_resized_images(dst_image_folders_base, resized_image_folders_base)

    ###############################
    # For an individual influencer:

    # influencer_filenames = influencers.get_influencer_files(
    #     src_image_folders_base, 'graceonyourdash')

    # influencer_subsampled_filenames = influencers.get_influencr_random_subsampled_files(
    #     influencer_filenames, 150)

    # influencers.copy_influencer_files(
    #     influencer_subsampled_filenames, src_image_folders_base, dst_image_folders_base)

    # influencers.save_resized_images(
    #     dst_image_folders_base, resized_image_folders_base)

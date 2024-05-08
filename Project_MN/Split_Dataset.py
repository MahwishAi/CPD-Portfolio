import os
import shutil
import random
import re

dataset_path = "grasp_dataset"

train_path = "train_data"
test_path = "test_data"

# Create train and test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

dataset_full_path = os.path.abspath(dataset_path)
class_folders = next(os.walk(dataset_full_path))[1]
print("Extracted folders:", class_folders)

for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        object_images = os.listdir(class_path)
        object_images.sort()  # Sort the images within each object folder

        # Extract the image number from the filename using regular expression
        image_numbers = [int(re.search(r'\((\d+)\)', image).group(1)) for image in object_images]

        # Calculate the number of images for training and testing
        num_images = len(object_images)
        num_train_images = int(num_images * 0.8)  # 80% for training, adjust as needed

        # Create class folders in the train and test directories
        train_class_path = os.path.join(train_path, class_folder)
        test_class_path = os.path.join(test_path, class_folder)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Randomly shuffle the image indices
        shuffled_indices = list(range(num_images))
        random.shuffle(shuffled_indices)

        # Split the indices into training and testing indices
        train_indices = shuffled_indices[:num_train_images]
        test_indices = shuffled_indices[num_train_images:]

        # Copy images to the train folder
        for index in train_indices:
            src_path = os.path.join(class_path, object_images[index])
            dst_path = os.path.join(train_class_path, object_images[index])
            shutil.copy(src_path, dst_path)

        # Copy images to the test folder
        for index in test_indices:
            src_path = os.path.join(class_path, object_images[index])
            dst_path = os.path.join(test_class_path, object_images[index])
            shutil.copy(src_path, dst_path)
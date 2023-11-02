import os
from pathlib import Path
import SimpleITK as sitk
from SimpleITK.SimpleITK import ProcessObject

# Paths for small_dataset
small_dataset = 'C://Users//Administrador//Documents//0. MAIA//3. Spain//4.MIRA//Lab2//trainingSmall//'

# Complete paths for training dataset, labels and masks
images_path = Path('C://Users//Administrador//Documents//0. MAIA//3. Spain//4.MIRA//Lab2//training_set//training-images//')
labels_path = Path('C://Users//Administrador//Documents//0. MAIA//3. Spain//4.MIRA//Lab2//training_set//training-labels//')
masks_path = Path('C://Users//Administrador//Documents//0. MAIA//3. Spain//4.MIRA//Lab2//training_set//training-labels//')
use_mask = False

# Bringing data paths
data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
masks_files = os.listdir(masks_path)

# Sorting files based on their name and extention
data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
masks_files.sort(key=lambda x: x.split('_')[0])

# Create a vector to store a collection of SimpleITK images
vectorOfImages = sitk.VectorOfImage()

# Read the first image (origin) from the data files
origin = sitk.ReadImage(os.path.join(images_path, data_files[0]))

# Loop through the data files
for filename in data_files:
    # Read each image file from the data files
    image = sitk.ReadImage(os.path.join(images_path, filename))
    
    # Resample the image to match the properties of the 'origin' image
    image = sitk.Resample(image, origin, sitk.Transform(), sitk.sitkBSpline, 0, origin.GetPixelID())
    
    # Add the resampled image to the vectorOfImages
    vectorOfImages.push_back(image)

# Combine the resampled images into a single 3D image (JoinSeries)
image = sitk.JoinSeries(vectorOfImages)

# Create an instance of the ElastixImageFilter for image registration
elastixImageFilter = sitk.ElastixImageFilter()

# Set the same image as both the fixed and moving images for groupwise registration
elastixImageFilter.SetFixedImage(image)
elastixImageFilter.SetMovingImage(image)

# Configure the parameter map for groupwise registration
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))

# Execute the groupwise image registration
image_group_wise = elastixImageFilter.Execute()

# Write the resulting groupwise registered image to a file
sitk.WriteImage(image_group_wise, "group.nii.gz")

print(1)


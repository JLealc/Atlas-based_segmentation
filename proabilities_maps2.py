import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

# Define the input directory containing labeled images for all patients
input_dir = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/training-set/training-set/training-labels'

# Define the output directory for probability maps
output_dir = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/probability_maps'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of patient subdirectories containing labeled images
labeled_image_files = os.listdir(input_dir)

# Define the labels for CSF, WM, and GM
CSF_label = 1
WM_label = 2
GM_label = 3

# Initialize lists to store tissue labels
CSF_labels = []
WM_labels = []
GM_labels = []

# List of patient indexes to process
train_indexes = labeled_image_files  # You can define this list with specific patient indexes

# Define a common shape for all binary masks
common_shape = (256, 256, 256)  # Adjust the shape as needed

# Iterate through labeled image files and create probability maps for each patient
for labeled_image_file in labeled_image_files:
    # Load the labeled image for the current patient
    labeled_image_path = os.path.join(input_dir, labeled_image_file)

    if os.path.exists(labeled_image_path):
        labeled_image = nib.load(labeled_image_path).get_fdata()

        # Initialize arrays to store labels for each index
        CSF_labels_per_index = []
        WM_labels_per_index = []
        GM_labels_per_index = []

        # Iterate through train_indexes
        for index in train_indexes:
            if index == labeled_image_file:  # Reference Image
                CSF = (labeled_image == CSF_label)
                WM = (labeled_image == WM_label)
                GM = (labeled_image == GM_label)
            else:  # Registered Training Images
                index_image_path = os.path.join(input_dir, index)
                index_image = nib.load(index_image_path).get_fdata()

                CSF = (index_image == CSF_label)
                WM = (index_image == WM_label)
                GM = (index_image == GM_label)

            # Resize or pad binary masks to the common shape without anti-aliasing
            CSF = resize(CSF, common_shape, anti_aliasing=False)
            WM = resize(WM, common_shape, anti_aliasing=False)
            GM = resize(GM, common_shape, anti_aliasing=False)

            # Append the binary masks to the respective lists
            CSF_labels_per_index.append(CSF)
            WM_labels_per_index.append(WM)
            GM_labels_per_index.append(GM)

        # Calculate probability atlases for each tissue type for the current patient
        prob_atlas_CSF = np.mean(CSF_labels_per_index, axis=0)
        prob_atlas_WM = np.mean(WM_labels_per_index, axis=0)
        prob_atlas_GM = np.mean(GM_labels_per_index, axis=0)

        # Append the probability atlases for the current patient to the main lists
        CSF_labels.append(prob_atlas_CSF)
        WM_labels.append(prob_atlas_WM)
        GM_labels.append(prob_atlas_GM)

# Calculate the final probability atlases
final_prob_atlas_CSF = np.mean(CSF_labels, axis=0)
final_prob_atlas_WM = np.mean(WM_labels, axis=0)
final_prob_atlas_GM = np.mean(GM_labels, axis=0)

# Load the fixed image to get its header information
fixed_image_path = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/training-set/training-set/training-labels/1000_3C.nii.gz'
fixed_image = nib.load(fixed_image_path)

# Define the output path for the final probability maps
output_path_CSF = os.path.join(output_dir, 'prob_atlas_CSF.nii.gz')
output_path_WM = os.path.join(output_dir, 'prob_atlas_WM.nii.gz')
output_path_GM = os.path.join(output_dir, 'prob_atlas_GM.nii.gz')

# Save the final probability maps with the header information of the fixed image
nib.save(nib.Nifti1Image(final_prob_atlas_CSF, fixed_image.affine), output_path_CSF)
nib.save(nib.Nifti1Image(final_prob_atlas_WM, fixed_image.affine), output_path_WM)
nib.save(nib.Nifti1Image(final_prob_atlas_GM, fixed_image.affine), output_path_GM)

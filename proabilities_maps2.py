import os
import numpy as np
import nibabel as nib

# Define the input directory containing labeled images for all patients
input_dir = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/training-set/training-set/training-labels'

# Define the output directory for probability maps
output_dir = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/atlas'

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

# Load the fixed image to get its header information
fixed_image_path = 'F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/training-set/training-set/training-labels/1000_3C.nii.gz'
fixed_image = nib.load(fixed_image_path)

# Variable to store the common shape for all binary masks
common_shape = None

for labeled_image_file in labeled_image_files:
    # Load the labeled image for the current patient
    labeled_image_path = os.path.join(input_dir, labeled_image_file)

    if os.path.exists(labeled_image_path):
        labeled_image = nib.load(labeled_image_path).get_fdata()

        # Create binary masks for CSF, WM, and GM labels
        CSF = (labeled_image == CSF_label)
        WM = (labeled_image == WM_label)
        GM = (labeled_image == GM_label)

        # If common_shape is not set, set it to the shape of the current binary mask
        if common_shape is None:
            common_shape = CSF.shape

        # Ensure all binary masks have the same shape
        CSF = CSF[:common_shape[0], :common_shape[1], :common_shape[2]]
        WM = WM[:common_shape[0], :common_shape[1], :common_shape[2]]
        GM = GM[:common_shape[0], :common_shape[1], :common_shape[2]]

        # Append the binary masks to the respective lists
        CSF_labels.append(CSF)
        WM_labels.append(WM)
        GM_labels.append(GM)

# Calculate probability atlases for each tissue type
prob_atlas_CSF = np.mean(CSF_labels, axis=0)
prob_atlas_WM = np.mean(WM_labels, axis=0)
prob_atlas_GM = np.mean(GM_labels, axis=0)

# Define the output path for the probability maps
output_path_CSF = os.path.join(output_dir, 'prob_atlas_CSF.nii')
output_path_WM = os.path.join(output_dir, 'prob_atlas_WM.nii')
output_path_GM = os.path.join(output_dir, 'prob_atlas_GM.nii')

# Save the probability maps with the header information of the fixed image
nib.save(nib.Nifti1Image(prob_atlas_CSF, fixed_image.affine), output_path_CSF)
nib.save(nib.Nifti1Image(prob_atlas_WM, fixed_image.affine), output_path_WM)
nib.save(nib.Nifti1Image(prob_atlas_GM, fixed_image.affine), output_path_GM)

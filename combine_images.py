import os
import numpy as np
import nibabel as nib

# Define the output directory for probability maps
output_dir = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results_labels'

# Create an empty array to accumulate the transformed images
combined_probabilities = None

# List of subject folders containing transformed images
subject_folders = os.listdir(output_dir)

for subject_folder in subject_folders:
    image_path = os.path.join(output_dir, subject_folder, 'result.nii')
    
    # Debug: Print the image_path for each subject folder
    print(f"Processing: {image_path}")
    
    if os.path.exists(image_path):
        img = nib.load(image_path)
        data = img.get_fdata()

        # Accumulate data into the combined_probabilities
        if combined_probabilities is None:
            combined_probabilities = data
        else:
            combined_probabilities += data
    else:
        # Debug: Print a message if the file doesn't exist
        print(f"File not found: {image_path}")

# Calculate the mean probability
mean_probability = combined_probabilities / len(subject_folders)

# Save the mean probability map
output_path = os.path.join(output_dir, 'mean_probability_map.nii.gz')
nib.save(nib.Nifti1Image(mean_probability, img.affine), output_path)

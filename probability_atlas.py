import os
import numpy as np
import nibabel as nib

# Specify the directory containing the probability maps
probability_maps_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\probability_maps_output1'

# Number of Gaussian components (tissue classes)
n_components = 3  # Change this based on your tissue classes (e.g., CSF, GM, WM)

# Initialize labels for tissue classes
tissue_labels = ['CSF', 'GM', 'WM']

# Create a dictionary to store normalized probability maps for each tissue class
normalized_probability_maps = {tissue: 0 for tissue in tissue_labels}

# Iterate through the probability maps for each tissue class
for tissue in tissue_labels:
    tissue_maps = []

    for subfolder in os.listdir(probability_maps_folder):
        if os.path.isdir(os.path.join(probability_maps_folder, subfolder)):
            # Load the probability map for the current tissue class
            probability_map_path = os.path.join(probability_maps_folder, subfolder, f'{subfolder}_{tissue}_probability_map.nii.gz')
            probability_map = nib.load(probability_map_path).get_fdata()
            tissue_maps.append(probability_map)

    # Accumulate the probability maps for the current tissue class
    total_map = np.sum(tissue_maps, axis=0)
    
    # Normalize the accumulated map
    normalized_map = total_map / np.sum(total_map)

    # Store the normalized map in the dictionary
    normalized_probability_maps[tissue] = normalized_map

# Combine the normalized probability maps into an average atlas
average_atlas = np.zeros_like(list(normalized_probability_maps.values())[0])

for tissue, normalized_map in normalized_probability_maps.items():
    average_atlas += normalized_map

# Save the average atlas as a NIfTI file
average_atlas_image = nib.Nifti1Image(average_atlas, nib.load(probability_map_path).affine)
output_path = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\average_atlas.nii.gz'
nib.save(average_atlas_image, output_path)

print(f"Average atlas saved to: {output_path}")

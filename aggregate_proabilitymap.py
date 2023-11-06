import os
import numpy as np
import nibabel as nib

# Define the input directory containing individual patient probability maps
input_dir = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\probability_maps'

# Define the output directory for the aggregated probability maps
output_dir = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\aggregated_probability_maps'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of tissue types (CSF, WM, GM)
tissue_types = ['CSF', 'WM', 'GM']

# Initialize dictionaries to store aggregated probability maps
aggregated_probability_maps = {tissue: None for tissue in tissue_types}

# List of patient probability map files
probability_map_files = os.listdir(input_dir)

# Iterate through each patient's probability maps
for probability_map_file in probability_map_files:
    # Extract patient ID and tissue type from the file name
    parts = os.path.splitext(os.path.basename(probability_map_file))[0].split('_')
    patient_id = parts[0]
    # Extract the tissue type correctly from the file name without the "_probability_map.nii.gz" part
    tissue_type = parts[-3]  # Get the third-to-last part of the filename

    input_path = os.path.join(input_dir, probability_map_file)

    if os.path.exists(input_path):
        probability_map = nib.load(input_path).get_fdata()
        print(f"Loaded: {input_path}, Patient ID: {patient_id}, Tissue Type: {tissue_type}")

        if tissue_type in aggregated_probability_maps:
            if aggregated_probability_maps[tissue_type] is None:
                aggregated_probability_maps[tissue_type] = probability_map
            else:
                # Ensure the input maps have the same shape
                if aggregated_probability_maps[tissue_type].shape == probability_map.shape:
                    aggregated_probability_maps[tissue_type] += probability_map
                else:
                    print(f"Warning: Incompatible dimensions for {input_path}. Skipping.")

# Calculate the mean probability map for each tissue type
for tissue_type in tissue_types:
    if aggregated_probability_maps[tissue_type] is not None:
        aggregated_probability_maps[tissue_type] /= len(probability_map_files)

        # Define the output path for the aggregated probability map
        output_path = os.path.join(output_dir, f'aggregated_{tissue_type}_probability_map.nii.gz')

        # Save the aggregated probability map
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(nib.Nifti1Image(aggregated_probability_maps[tissue_type], nib.load(input_path).affine), output_path)
        print(f"Saved: {output_path}")

print("Aggregation completed.")

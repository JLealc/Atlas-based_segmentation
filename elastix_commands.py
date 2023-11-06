import os
import subprocess
import numpy as np
import nibabel as nib

# Specify the directory containing training images
input_image_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\training-set\\training-set\\training-images\\'

# Specify the directory containing corresponding label images
input_label_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\training-set\\training-set\\training-labels\\'

# Specify the fixed image (you may also choose this from the input folder)
fixed_image = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\training-set\\training-set\\training-images\\1000.nii.gz'

# Create an output folder to store the results
output_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results'
os.makedirs(output_folder, exist_ok=True)

output_folder_labels = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results_labels'
os.makedirs(output_folder_labels, exist_ok=True)

os.chdir('F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\elastix-5.0.0-win64\\')
elastix_path = 'elastix.exe'

os.chdir('F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\elastix-5.0.0-win64\\')
transformix_path = 'transformix.exe'

# Get a list of image files in the input image folder
image_files = [os.path.join(input_image_folder, filename) for filename in os.listdir(input_image_folder) if filename.endswith('.nii.gz')]

# Create a dictionary to store elastix and transformix results for each image
registration_results = {}

# Specify the location of the parameter files (outside of the patient folders)
parameter_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\'
parameter_affine = os.path.join(parameter_folder, 'Parameters_Affine.txt')
parameter_bspline = os.path.join(parameter_folder, 'Parameters_BSpline.txt')

# Loop through each image file and apply elastix and transformix
for image_file in image_files:
    # Extract the image file name (without extension) to construct output names
    image_base_name = os.path.splitext(os.path.splitext(os.path.basename(image_file))[0])[0]

    # Create the output directory for the current image
    output_image_folder = os.path.join(output_folder, image_base_name)
    os.makedirs(output_image_folder, exist_ok=True)

    # Check if elastix and transformix results already exist for this image
    if image_base_name in registration_results:
        print(f"Using existing results for {image_base_name}")
    else:
        # Check if the flag file exists; if not, process the image
        flag_file_path = os.path.join(output_folder, image_base_name + '.processed')
        if os.path.exists(flag_file_path):
            print(f"Skipping {image_base_name} as it has already been processed.")
        else:
            # Step 2: Register the current image to the fixed image using elastix
            elastix_command = [
                elastix_path,
                '-f', fixed_image,
                '-m', image_file,
                '-p', parameter_affine,
                '-p', parameter_bspline,
                '-out', output_image_folder
            ]
            subprocess.run(elastix_command)

            # Step 3: Transform the current image using transformix
            transformix_command = [
                transformix_path,
                '-in', image_file,
                '-out', output_image_folder,
                '-tp', os.path.join(output_image_folder, 'TransformParameters.0.txt')
            ]
            subprocess.run(transformix_command)

            # Create the output directory for the current labels
            output_label_folder = os.path.join(output_folder_labels, image_base_name)
            os.makedirs(output_label_folder, exist_ok=True)

            # Construct the correct label file name based on your naming pattern
            label_file = os.path.join(input_label_folder, f'{image_base_name}_3C.nii.gz')

            # Modify the parameter file for label transformation
            label_parameter_file = os.path.join(output_image_folder, 'TransformParameters.1.txt')

            with open(label_parameter_file, 'r') as file:
                label_parameter_lines = file.readlines()

            for i, line in enumerate(label_parameter_lines):
                if line.startswith('(FinalBSplineInterpolationOrder'):
                    label_parameter_lines[i] = '(FinalBSplineInterpolationOrder 0)\n'

            with open(label_parameter_file, 'w') as file:
                file.writelines(label_parameter_lines)

            transformix_command_labels = [
                transformix_path,
                '-in', label_file,
                '-out', output_label_folder,
                '-tp', label_parameter_file
            ]
            subprocess.run(transformix_command_labels)

            # Store the results in the dictionary
            registration_results[image_base_name] = output_image_folder

            # Create the flag file to mark this image as processed
            open(flag_file_path, 'w').close()

print("Processing completed.")

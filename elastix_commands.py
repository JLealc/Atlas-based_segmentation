import os
import subprocess

# -------------------------------------------------------------------------------
# STEP 1: Registration
# Paths
input_image_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/training-set/training-set/training-images/'
fixed_image = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/training-set/training-set/training-images/1000.nii.gz'
output_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/registration_results'
affine_reg = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/affine_reg.txt'
bspline_reg = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/bspline_reg.txt'
os.makedirs(output_folder, exist_ok=True)

# Elastix configuration
os.chdir('C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/elastix-5.0.0-win64/')
elastix_path = 'elastix.exe'

# Images for which registration needs to be done
image_ids = [1001,1002, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1017, 1036]

# Construct the elastix command for each image
for image_id in image_ids:
    image_output_folder = os.path.join(output_folder, f"{str(image_id).zfill(2)}")
    os.makedirs(image_output_folder, exist_ok=True)  # Create the directory if it doesn't exist
    
    command = [
        elastix_path, 
        "-f", fixed_image,
        "-m", os.path.join(input_image_folder, f"{image_id}.nii.gz"),
        "-p", affine_reg, 
        "-P", bspline_reg,
        "-out", image_output_folder + "/"
    ]
    
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Print the output and any errors
    print(stdout.decode())
    if stderr:
        print(f"Errors for image {image_id}:", stderr.decode())

# ---------------------------------------------------------------------------------------------------------------------
# STEP 2: Transformation

# Paths
training_labels_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/training-set/training-set/training-labels/'
registered_images_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/registration_results/'
transformed_labels_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/transformed_labels/'

# Ensure the base output folder for the transformed labels exists
os.makedirs(transformed_labels_folder, exist_ok=True)

# Elastix configuration
os.chdir('C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/elastix-5.0.0-win64/')
transformix_path = 'transformix.exe'

# List of image IDs
image_ids = [1001, 1002, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1017, 1036]

# Generate and run transformix command for each image
for image_id in image_ids:
    # Create the specific folder for the transformed labels
    transformed_label_output_folder = os.path.join(transformed_labels_folder, f"{str(image_id).zfill(2)}")
    os.makedirs(transformed_label_output_folder, exist_ok=True)

    # Setup the paths
    input_label_path = os.path.join(training_labels_folder, f"{image_id}_3C.nii.gz")
    transform_param_file = os.path.join(registered_images_folder, f"{str(image_id).zfill(2)}/", "TransformParameters.1.txt")
    output_path = transformed_label_output_folder + "/"

    # Construct the transformix command
    command = [
        transformix_path,
        "-in", input_label_path,
        "-tp", transform_param_file,
        "-out", output_path
    ]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Print the output and any errors
    print(stdout.decode())
    if stderr:
        print(f"Errors for image {image_id}:", stderr.decode())

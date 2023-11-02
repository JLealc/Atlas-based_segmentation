import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk

# Paths for small_dataset
small_dataset = 'C://Users//Administrador//Documents//0. MAIA//3. Spain//4.MIRA//Lab2//trainingSmall//'
def word():
    print(18)

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

# Registration with Similarity='SetMetricAsMattesMutualInformation'
Reg_MI = sitk.ImageRegistrationMethod()
Reg_MI.SetMetricAsMattesMutualInformation()

# Registration with Similarity='SetMetricAsMeanSquares'
Reg_MSE = sitk.ImageRegistrationMethod()
Reg_MSE.SetMetricAsMeanSquares()

# use each image as the fixed image and pair it and check the csv file for the best image
# use mask = True
# make group wise segmentation and check for all pairs MI and mean values and compare
# register with the image masks
df_values_list = []

#############Metric Value###interpolator########

# label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
# label_shape_filter.Execute(moving_mask)
# bounding_box = label_shape_filter.GetBoundingBox(1)

# roi_filter=sitk.RegionOfInterestImageFilter()

# fixed_image = sitk.ReadImage(os.path.join(images_path, data_files[0]))
# fixed_mask = sitk.ReadImage(os.path.join(labels_path, labels_files[0]), sitk.sitkUInt8)

# Loop through pairs of fixed and label files
for data_fixed, labels_fixed in zip(data_files, labels_files):
    
    # Load the fixed image and its corresponding label 
    fixed_image = sitk.ReadImage(os.path.join(images_path, data_fixed))
    fixed_mask = sitk.ReadImage(os.path.join(labels_path, labels_fixed), sitk.sitkUInt8)

    # Loop through pairs of moving and label files
    for data_moving, label_moving in zip(data_files, labels_files):
        # Skip processing if the moving image is the same as the fixed image
        if data_moving == data_fixed:
            continue

        # Print information about the fixed and moving images being processed
        print(f'fixed image {data_fixed}, moving image {data_moving}')

        # Define an output folder for the results
        out_folder = Path(f'./{use_mask}/{data_fixed}/{data_moving}')

        # Create the output folder if it doesn't exist
        out_folder.mkdir(parents=True, exist_ok=True)

        # Initialize a dictionary to store results and parameters
        df_values = {}
        df_values['fixed_image'] = data_fixed
        df_values['moving_image'] = data_moving
        df_values['use_mask'] = use_mask

        # Load the moving image
        moving_image = sitk.ReadImage(os.path.join(images_path, data_moving))

        # Create an instance of the ElastixImageFilter for image registration
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        # If use_mask is True, load and set masks for the fixed and moving images
        if use_mask:
            moving_image_mask = sitk.ReadImage(os.path.join(labels_path, label_moving), sitk.sitkUInt8)
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)

        # Set the parameter map for the image registration (rigid transformation in this case)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))

        # Record the start time for measuring execution time
        start_time = time.time()

        # Execute the image registration (rigid transformation)
        rigid_out_img = elastixImageFilter.Execute()

        # Print the time taken for the rigid transformation
        print(f'rigid took {(time.time() - start_time)}')

        # Evaluate the Mutual Information and Mean Squares Error metrics for the rigid transformation
        rigid_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, rigid_out_img)
        rigid_mse_value = Reg_MSE.MetricEvaluate(fixed_image, rigid_out_img)

        # Store the metrics and parameters in the df_values dictionary
        df_values['rigid_mi'] = rigid_mutual_info_metric_value
        df_values['rigid_mse'] = rigid_mse_value

        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], str(
        #     out_folder / Path('rigid_transform.txt')))

        # ========== Affine ==========
        
        # Create an instance of the ElastixImageFilter for image registration
        elastixImageFilter = sitk.ElastixImageFilter()

        # Set the fixed and moving images for registration
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        # If 'use_mask' is True, set the fixed and moving masks for registration
        if use_mask:
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)

        # Get the default parameter map for affine transformation
        parameterMap = sitk.GetDefaultParameterMap('affine')

        # Enable masking during the registration process
        parameterMap["fMask"] = ["true"]
        parameterMap["ErodeMask"] = ["true"]

        # Set the parameter map for the image registration to 'affine'
        elastixImageFilter.SetParameterMap(parameterMap)

        # Record the start time for measuring execution time
        start_time = time.time()

        # Execute the image registration (affine transformation)
        affine_out_img = elastixImageFilter.Execute()

        # Print the time taken for the affine transformation
        print(f'affine took {(time.time() - start_time)}')

        # sitk.WriteImage(elastixImageFilter.GetResultImage())
        affine_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, affine_out_img)
        affine_mse_value = Reg_MSE.MetricEvaluate(fixed_image, affine_out_img)
        df_values['affine_mi'] = affine_mutual_info_metric_value
        df_values['affine_mse'] = affine_mse_value

        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0],
        #                         str(out_folder / Path('affine_transform.txt')))

        # ========== Non Rigid Registration ==========
        # Create an instance of the ElastixImageFilter for non-rigid (bspline) image registration
        elastixImageFilter = sitk.ElastixImageFilter()

        # Set the fixed and moving images for registration
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        # If 'use_mask' is True, set the fixed and moving masks for registration
        if use_mask:
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)

        # Create a vector to hold multiple parameter maps
        parameterMapVector = sitk.VectorOfParameterMap()

        # Append default parameter maps for affine and bspline transformations
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))

        # Set the parameter map vector for the image registration, allowing both affine and bspline transformations
        elastixImageFilter.SetParameterMap(parameterMapVector)

        # Record the start time for measuring execution time
        start_time = time.time()

        # Execute the image registration with both affine and bspline transformations
        non_rigid_out_img = elastixImageFilter.Execute()

        # Print the time taken for the bspline transformation
        print(f'bspline took {(time.time() - start_time)}')

        # sitk.WriteImage(elastixImageFilter.GetResultImage())
        non_rigid_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, non_rigid_out_img)
        non_rigid_mse_value = Reg_MSE.MetricEvaluate(fixed_image, non_rigid_out_img)
        df_values['non_rigid_mi'] = non_rigid_mutual_info_metric_value
        df_values['non_rigid_mse'] = non_rigid_mse_value
        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0],
        #                         str(out_folder / Path('non_rigid_transform.txt')))
        df_values_list.append(df_values)
        # exit()
# df = pd.DataFrame(df_values_list)
# df.to_csv('reg_results_2.csv')
print(1)

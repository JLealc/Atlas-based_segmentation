import os
import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Specify the directory containing registration results
registered_image_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results'

# Specify the directory containing transformed labels
labels_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results_labels'

output_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\soft_probability_maps'

# Number of Gaussian components (clusters)
n_components = 3  # Change this based on your tissue classes (e.g., CSF, GM, WM)

# Initialize labels for tissue classes
tissue_labels = ['CSF', 'GM', 'WM']

# Iterate through the subdirectories in the registration_results folder
for subfolder in os.listdir(registered_image_folder):
    subfolder_path = os.path.join(registered_image_folder, subfolder)

    if os.path.isdir(subfolder_path):
        # Load the result of the registration image
        image_path = os.path.join(subfolder_path, 'result.nii')
        image = nib.load(image_path)
        image_data = image.get_fdata()

        # Load the transformed label as a mask
        label_path = os.path.join(labels_folder, subfolder, 'result.nii')
        label_image = nib.load(label_path)
        label_data = label_image.get_fdata()

        # Create a probability map for each tissue class
        probability_maps = []

        for i in range(n_components):
            # Use the transformed label as a mask and exclude zero intensities
            mask = (label_data == (i + 1))  # Adjust for label values
            image_data_masked = image_data[mask]

            # Reshape the image data for clustering
            image_data_masked = image_data_masked.reshape(-1, 1)

            # Fit a Gaussian Mixture Model with K-Means initialization
            gmm = GaussianMixture(n_components=1, covariance_type='spherical', init_params='kmeans', random_state=0)
            gmm.fit(image_data_masked)

            # Generate tissue probability map
            predicted_probabilities = gmm.predict_proba(image_data_masked)
            probability_map = np.zeros(mask.shape)
            probability_map[mask] = predicted_probabilities[:, 0]

            probability_maps.append(probability_map)

            # Create a scatter plot for the clusters
            for cluster_id in range(gmm.n_components):
                cluster_indices = np.where(gmm.predict(image_data_masked) == cluster_id)
                cluster_data = image_data_masked[cluster_indices]
                plt.scatter(cluster_data, np.zeros_like(cluster_data), label=f'Cluster {cluster_id + 1}')
                plt.legend()
                plt.title(f'Cluster Plot for {subfolder}')
                plt.show()

        # You can save these probability maps as NIfTI files if needed:
        for i, tissue in enumerate(tissue_labels):
            tissue_map = probability_maps[i]
            probability_image = nib.Nifti1Image(tissue_map, image.affine)
            output_path = os.path.join(output_folder, f'{subfolder}_{tissue}_probability_map.nii.gz')
            nib.save(probability_image, output_path)
            print(f"Saved: {output_path}")

print("Processing completed.")


# import os
# import numpy as np
# import nibabel as nib
# from sklearn.mixture import GaussianMixture

# # Load the result of registration image
# image_path = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results\\1001.nii\\result.nii'
# image = nib.load(image_path)
# image_data = image.get_fdata()

# # Load the transformed label as a mask
# label_path = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results\\1001\\result.nii'
# label_image = nib.load(label_path)
# label_data = label_image.get_fdata()

# # Use the transformed label as a mask and exclude zero intensities
# mask = (label_data > 0)
# image_data = image_data[mask]

# # Reshape the image data for clustering
# image_data = image_data.reshape(-1, 1)

# # Number of Gaussian components (clusters)
# n_components = 3  # Change this based on your tissue classes (e.g., CSF, WM, GM)

# # Fit a Gaussian Mixture Model with K-Means initialization
# gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', init_params='kmeans', random_state=0)
# gmm.fit(image_data)

# # Generate tissue probability maps
# predicted_probabilities = gmm.predict_proba(image_data)
# probability_maps = np.zeros(mask.shape + (n_components,))
# probability_maps[mask] = predicted_probabilities

# # You can save these probability maps as NIfTI files if needed:
# output_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\probability_maps_output'
# os.makedirs(output_folder, exist_ok=True)

# for i, tissue in enumerate(['CSF', 'GM', 'WM']):
#     tissue_map = probability_maps[..., i]
#     probability_image = nib.Nifti1Image(tissue_map, image.affine)
#     output_path = os.path.join(output_folder, f'1001_{tissue}_probability_map.nii.gz')
#     nib.save(probability_image, output_path)
#     print(f"Saved: {output_path}")

# print("Processing completed.")
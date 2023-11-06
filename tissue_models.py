import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Specify the directory containing registered images
registered_image_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results'

# Specify the directory containing transformed labels
labels_folder = 'F:\\MAIA\\third semester\\4. MIRA\\Atlas-based_segmentation\\registration_results_labels'

# Initialize labels for tissue classes
tissue_labels = ['CSF', 'GM', 'WM']

# Get a list of subfolders
subfolders = [subfolder for subfolder in os.listdir(registered_image_folder) if os.path.isdir(os.path.join(registered_image_folder, subfolder))]

# Calculate the number of rows and columns for subplots
num_subfolders = len(subfolders)
num_cols = 4  # Set the number of columns in the subplot grid
num_rows = ((num_subfolders + num_cols - 1) // num_cols)  # Calculate the number of rows needed

# Create a figure with subplots for all tissue density models
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

# Flatten the axs array if it's a single row or column
if num_rows == 1:
    axs = axs[None, :]

if num_cols == 1:
    axs = axs[:, None]

# Create lists to store tissue density models
tissue_density_models = {tissue: [] for tissue in tissue_labels}

# Create a list of line styles for different tissues
line_styles = ['-', '--', ':']

# Iterate through the subdirectories in the registration_results folder
for subfolder_index, subfolder in enumerate(subfolders):
    subfolder_path = os.path.join(registered_image_folder, subfolder)

    # Load the registered image
    image_path = os.path.join(subfolder_path, 'result.nii')
    image = nib.load(image_path)
    image_data = image.get_fdata()

    # Load the transformed label as a mask
    label_path = os.path.join(labels_folder, subfolder, 'result.nii')
    label_image = nib.load(label_path)
    label_data = label_image.get_fdata()

    # Create a range of values for plotting the density
    x_range = np.linspace(0, 255, 1000)  # Adjust the range as needed
    log_dens = []

    # Iterate through tissue classes and create density models
    for i, tissue in enumerate(tissue_labels):
        # Use the transformed label as a mask and exclude zero intensities
        mask = (label_data == (i + 1))  # Adjust for label values
        tissue_data = image_data[mask]

        # Create a density model using Kernel Density Estimation (KDE)
        kde = KernelDensity(bandwidth=5.0, kernel='gaussian')  # Adjust the bandwidth as needed
        kde.fit(tissue_data[:, np.newaxis])

        # Generate density estimates for the x_range
        log_density = kde.score_samples(x_range[:, np.newaxis])
        log_dens.append(log_density)

        # Normalize the density estimates with a small constant (epsilon) to avoid divide by zero
        epsilon = 1e-10  # Small constant to prevent division by zero
        integral = np.trapz(np.exp(log_density), x_range)
        log_density -= np.log(integral + epsilon)

        # Calculate the row and column index for the current subplot
        row = subfolder_index // num_cols
        col = subfolder_index % num_cols

        # Plot the tissue density model in the current subplot with a different line style and fill the area
        line_style = line_styles[i]
        axs[row, col].plot(x_range, np.exp(log_density), label=f'{tissue}', linestyle=line_style)
        axs[row, col].fill_between(x_range, 0, np.exp(log_density), alpha=0.2)
        axs[row, col].set_title(subfolder)


# Set common labels and show the plots
for ax in axs.flat:
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

print("Tissue density models computed and visualized.")

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from atlas_tissues import atlas_probabilities, tissue_models

# Paths
atlas_path = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/atlas/'
tissue_model_paths = [f'{atlas_path}tissueModel_CSF.npy', f'{atlas_path}tissueModel_WM.npy', f'{atlas_path}tissueModel_GM.npy']

# ---------------------------------------------------------------------------------------------------------------------
# STEP 1: Generate Probability Atlas
print("Executing label propagation via probability atlases...")
atlas_probabilities(export_option='return')
print("Label propagation via probability atlases completed.")

# ---------------------------------------------------------------------------------------------------------------------
# STEP 2: Generate Tissue Models
print("Executing label propagation via probability atlases and tissue models...")
tissue_models(export_option='save')
print("Label propagation via probability atlases and tissue models completed.")

# ---------------------------------------------------------------------------------------------------------------------
# STEP 3: Load Tissue Models for plotting
print("Loading tissue models...")
# Load Tissue Models
CSF, WM, GM = [np.load(path) for path in tissue_model_paths]
print("Tissue models loaded.")

# ---------------------------------------------------------------------------------------------------------------------
# STEP 4: Visualize independent tissue models
print("Visualizing independent tissue models...")
plt.figure()
hist_params = {'bins': 2000, 'range': (0, 2000), 'density': True, 'alpha': 0.75}

for tissue, color in zip([CSF, WM, GM], ['red', 'green', 'blue']):
    plt.hist(tissue, **hist_params, color=color)

plt.title('Independent Tissue Models')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')
plt.legend(['CSF', 'WM', 'GM'], loc='upper right', fontsize='x-large')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# STEP 5: Generate and visualize plottings for functional tissue models
print("Visualizing functional tissue models...")
suitable_max = np.percentile(np.concatenate((CSF, WM, GM)), 90)
hist_params = {'bins': 2000, 'range': (0, 2000)}
iTP_data = [np.histogram(tissue, **hist_params)[0] for tissue in [CSF, WM, GM]]
hist_sum = np.sum(iTP_data, axis=0)
iTP_normalized = [np.nan_to_num(hist / hist_sum) for hist in iTP_data]

plt.figure()

# Use the same bins for all tissues
bins = np.histogram(CSF, **hist_params)[1]
bin_width = bins[1] - bins[0]

for tissue, color in zip(iTP_normalized, ['red', 'green', 'blue']):
    plt.bar(bins[:-1], tissue, width=bin_width, alpha=0.75, color=color)

plt.xlabel('Intensity Values')
plt.ylabel('Probability')
plt.legend(['CSF', 'WM', 'GM'], loc='upper right')
plt.show()

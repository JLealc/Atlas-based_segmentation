# Libraries
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from prime_aux import computeAtlasProb, computeTissueModels, labelPropg, all_labelPropg

#EXECUTE ONE-TIME TO STORE ORIGINALS
# -------------------------------------------------------------------------------

print("Executing label propagation via probability atlases...")
### PROBABILITY ATLASES
computeAtlasProb(export='save')
print("Label propagation via probability atlases completed.")

# -------------------------------------------------------------------------------

print("Executing label propagation via probability atlases and tissue models...")
computeTissueModels(export='save')
print("Label propagation via probability atlases and tissue models completed.")

# ----------------------------------------------------------------------------------
### LABEL PROPAGATION VIA PROBABILITY ATLASES
#all_labelPropg(mode='prob_atlas')

### LABEL PROPAGATION VIA PROBABILITY ATLASES + TISSUE MODELS
#all_labelPropg(mode='prob_inten_atlas')'''
# -------------------------------------------------------------------------------
'''
print("Starting to generate probabilistic atlases from training data...")
# Generate Probabilistic Atlases from Training Data
prob_atlas_CSF, prob_atlas_WM, prob_atlas_GM  = computeAtlasProb(export='save')
print("Probabilistic atlases generated.")'''

print("Loading tissue models...")
# Load Tissue Models
CSF = np.load('F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/atlas/tissueModel_CSF.npy')
WM  = np.load('F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/atlas/tissueModel_WM.npy')
GM  = np.load('F:/MAIA/third semester/4. MIRA/Atlas-based_segmentation/atlas/tissueModel_GM.npy')
print("Tissue models loaded.")
#----------------------------------------------------------------------------------------------------------

print("Visualizing independent tissue models...")
# Parameters for all histograms
hist_params = {'bins': 2000, 'range': (0, 2000), 'density': True, 'alpha': 0.75}

# Plot histograms for CSF, WM, and GM with shared parameters and updated colors
plt.figure()
plt.hist(CSF, **hist_params, label='CSF', color='red')
plt.hist(WM,  **hist_params, label='WM',  color='green')
plt.hist(GM,  **hist_params, label='GM',  color='blue')

# Set titles and labels
plt.title('Independent Tissue Models')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')

# Set legend
plt.legend(loc='upper right', fontsize='x-large')

# Display the plot
plt.show()

#----------------------------------------------------------------------------------------------------------
print("Visualizing functional tissue models...")
# Visualize Functional Tissue Models
suitable_max = np.percentile(np.concatenate((CSF, WM, GM)), 90)
# Compute histograms for each tissue type with shared parameters
hist_params = {'bins': 2000, 'range': (0, 2000)}
iTP_CSF, bins = np.histogram(CSF, **hist_params)
iTP_WM, _ = np.histogram(WM, **hist_params)
iTP_GM, _ = np.histogram(GM, **hist_params)

# Pre-calculate the sum for normalization to avoid redundancy
hist_sum = iTP_CSF + iTP_WM + iTP_GM

# Normalize histograms to obtain probabilities
CSF_bins = np.nan_to_num(iTP_CSF / hist_sum)
WM_bins = np.nan_to_num(iTP_WM / hist_sum)
GM_bins = np.nan_to_num(iTP_GM / hist_sum)

# Functional Tissue Probabilities
fTP_CSF = (CSF_bins, bins[:-1])
fTP_WM = (WM_bins, bins[:-1])
fTP_GM = (GM_bins, bins[:-1])

# Visualize Functional Tissue Models
plt.figure()

# Use width corresponding to bin width for accurate visualization
bin_width = bins[1] - bins[0]

# Plot bar graphs for each tissue type with shared parameters and updated colors
plt.bar(fTP_CSF[1], fTP_CSF[0], width=bin_width, alpha=0.75, label='CSF', color='red')
plt.bar(fTP_WM[1], fTP_WM[0], width=bin_width, alpha=0.75, label='WM', color='green')
plt.bar(fTP_GM[1], fTP_GM[0], width=bin_width, alpha=0.75, label='GM', color='blue')

# Set titles and labels
plt.xlabel('Intensity Values')
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.show()

print("Preparing for segmentation...")
# Segmentation:
index = '038'

'''
# Segmentation via Training Image-Based Probabilistic Atlas
predicted_mask00 = labelPropg(CSF="../results/testing_results/transformed_labels/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode='prob_atlas', export='return')

# Segmentation via MNI Probabilistic Atlas
predicted_mask01 = labelPropg(CSF="../results/testing_results/transformed_labels_MNI/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels_MNI/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_atlas", export='return')


# Segmentation via Training Image-Based Probability Atlas
predicted_mask10 = labelPropg(CSF="../results/testing_results/transformed_labels/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_inten_atlas", export="return")

# Segmentation via MNI Probabilistic Atlas
predicted_mask11 = labelPropg(CSF="../results/testing_results/transformed_labels_MNI/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels_MNI/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_inten_atlas", export="return")

# Segmentation via Expectation-Maximization with K-Means Initialization
mask1, score1   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="kmeans", atlas=None, mode="base", export="return")

# Segmentation via Expectation-Maximization with Probabilistic Atlas Initialization
mask2, score2   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='training', mode="base", export="return")

# Segmentation via Expectation-Maximization with Probabilistic Atlas Initialization and Late Fusion
mask3, score3   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='training', mode="atlas", export="return")

# Segmentation via Expectation-Maximization with MNI Atlas Initialization
mask4, score4   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='MNI', mode="base", export="return")

# Segmentation via Expectation-Maximization with MNI Atlas Initialization and Late Fusion
mask5, score5   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='MNI', mode="atlas", export="return")
'''


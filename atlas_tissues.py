import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def create_masks(label_image_path, volume_image_path):
    # Load label and volume images as numpy arrays
    labels = sitk.GetArrayFromImage(sitk.ReadImage(label_image_path))
    volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_image_path, sitk.sitkFloat32))

    # Initialize masks for CSF, WM, GM
    masks = [np.zeros_like(labels) for _ in range(3)]
    
    # Define tissue types with corresponding label values
    tissue_types = [1, 2, 3]  # 1 for CSF, 2 for WM, 3 for GM

    # Apply masks to the volume for each tissue type
    masked_volumes = []
    for i, tissue in enumerate(tissue_types):
        masks[i][labels == tissue] = 1
        masked_volumes.append(volume * masks[i].astype(bool))
    
    return masked_volumes

def extract_non_zero_features(label_image_path, volume_image_path): 
    # Decompose volumes by individual masks
    masked_volumes = create_masks(label_image_path, volume_image_path)

    # Extract non-zero feature vectors efficiently
    non_zero_features = [masked_volume[masked_volume.nonzero()].reshape(-1, 1) for masked_volume in masked_volumes]
    
    return non_zero_features

def tissue_models(export_option):
    # List of indexes for training
    train_indexes = ['000', '001', '002', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '017', '036']
    
    # Base paths for labels and images
    base_label_path = "C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/training-set/training-set/training-labels/1"
    base_image_path = "C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/training-set/training-set/training-images/1"

    # Initialize tissue models as None
    csf_tissue_model, wm_tissue_model, gm_tissue_model = None, None, None

    # Process each index
    for index in tqdm(train_indexes, desc="Computing Tissue Models"):
        label_path = f"{base_label_path}{index}_3C.nii.gz"
        image_path = f"{base_image_path}{index}.nii.gz"
        
        # Extract non-zero features for CSF, WM, GM
        csf, wm, gm = extract_non_zero_features(label_path, image_path)
        
        # Concatenate the new data to the existing arrays
        if csf_tissue_model is not None:
            csf_tissue_model = np.concatenate((csf_tissue_model, csf), axis=0)
        else:
            csf_tissue_model = csf
        
        if wm_tissue_model is not None:
            wm_tissue_model = np.concatenate((wm_tissue_model, wm), axis=0)
        else:
            wm_tissue_model = wm
        
        if gm_tissue_model is not None:
            gm_tissue_model = np.concatenate((gm_tissue_model, gm), axis=0)
        else:
            gm_tissue_model = gm

    # Handle saving or returning of models based on the 'export_option' parameter
    tissue_models = {'CSF': csf_tissue_model, 'WM': wm_tissue_model, 'GM': gm_tissue_model}
    atlas_path = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation/atlas/'
    
    if export_option == 'save':
        for tissue, model in tissue_models.items():
            np.save(f'{atlas_path}tissueModel_{tissue}.npy', model)
    elif export_option == 'return':
        return csf_tissue_model, wm_tissue_model, gm_tissue_model

def atlas_probabilities(export_option):
    train_indexes = ['000', '001', '002', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '017', '036']
    label_path = "C:/Users/Administrador/Documents/0. MAIA/3. Spain/4.MIRA/Lab2/Atlas-based_segmentation"
    reference_image_path = f"{label_path}/training-set/training-set/training-labels/1000_3C.nii.gz"
    reference_image = sitk.ReadImage(reference_image_path)
    label_shape = sitk.GetArrayFromImage(reference_image).shape
    num_labels = len(train_indexes)

    csf_labels = np.zeros(label_shape + (num_labels,))
    wm_labels = np.zeros_like(csf_labels)
    gm_labels = np.zeros_like(csf_labels)

    for i, index in enumerate(train_indexes):
        if i == 0:  # Reference Image
            file_path = f"{label_path}/training-set/training-set/training-labels/1{index}_3C.nii.gz"
        else:  # Registered Training Images
            file_path = f"{label_path}/transformed_labels/1{index}/result.mhd"
        
        label = np.array(sitk.GetArrayFromImage(sitk.ReadImage(file_path)))
        csf_labels[:, :, :, i] = (label == 1)
        wm_labels[:, :, :, i]  = (label == 2)
        gm_labels[:, :, :, i]  = (label == 3)

    # Compute Probability Atlases for Each Label
    prob_atlas_csf = np.mean(csf_labels, axis=3)
    prob_atlas_wm = np.mean(wm_labels, axis=3)
    prob_atlas_gm = np.mean(gm_labels, axis=3)

    if export_option == 'save':
        # Set up the writer once
        writer = sitk.ImageFileWriter()
        output_paths = [
            f"{label_path}/atlas/prob_atlas_csf.nii",
            f"{label_path}/atlas/prob_atlas_wm.nii",
            f"{label_path}/atlas/prob_atlas_gm.nii"
        ]
        
        for prob_atlas, output_path in zip([prob_atlas_csf, prob_atlas_wm, prob_atlas_gm], output_paths):
            output_image = sitk.GetImageFromArray(prob_atlas)
            output_image.CopyInformation(reference_image)
            writer.SetFileName(output_path)
            writer.Execute(output_image)
        
    elif export_option == 'return':
        return prob_atlas_csf, prob_atlas_wm, prob_atlas_gm

def create_combined_atlas(csf_atlas_path, wm_atlas_path, gm_atlas_path, combined_atlas_path):
    # Read the atlas images and remove any singleton dimensions
    csf_atlas = sitk.GetArrayFromImage(sitk.ReadImage(csf_atlas_path)).squeeze()
    wm_atlas = sitk.GetArrayFromImage(sitk.ReadImage(wm_atlas_path)).squeeze()
    gm_atlas = sitk.GetArrayFromImage(sitk.ReadImage(gm_atlas_path)).squeeze()

    # Check if all atlas images have the same shape after squeezing
    if not (csf_atlas.shape == wm_atlas.shape == gm_atlas.shape):
        raise ValueError("The input atlas images must have the same dimensions.")

    # Initialize the combined atlas
    combined_atlas = np.zeros_like(csf_atlas)

    # Fill the combined atlas with weighted values (You may change these as needed)
    combined_atlas += csf_atlas * 1  # Weight for CSF
    combined_atlas += wm_atlas * 2  # Weight for WM
    combined_atlas += gm_atlas * 3  # Weight for GM

    # Normalize the combined atlas if needed (for example, to a range of 0-255)
    combined_atlas = ((combined_atlas - combined_atlas.min()) * (255.0 / (combined_atlas.max() - combined_atlas.min()))).astype(np.uint8)

    # Convert the numpy array back to a SimpleITK Image
    combined_atlas_image = sitk.GetImageFromArray(combined_atlas.astype(np.uint8))

    # Copy the metadata from the reference image
    reference_image = sitk.ReadImage(csf_atlas_path)
    combined_atlas_image.CopyInformation(reference_image)

    # Save the combined atlas
    sitk.WriteImage(combined_atlas_image, combined_atlas_path)


'''
Representational Similarity Analysis (RSA)

Performs RSA between the earlier computed beta maps and the contextual word embeddings for each of the 
672 words and for each of the 39 ROIs in our Neurosynth meta-analytic parcellation. This parcellation 
is based on the FDR-corrected significant positive effects from the MKDA meta-analysis of the top topics related to emotional 
concept representation.

WORK IN PROGRESS: CURRENTLY COMPUTING SIMILARITY MATRICES FOR ALL SUBJECTS AND ALL ROIs, THEN SAVING TO DISK.
'''

#####################################################################
# Imports
#####################################################################
import os
import glob
from nilearn import image, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltools.data import Adjacency
from sklearn.metrics import pairwise_distances
from bids import BIDSLayout
from nilearn.maskers import NiftiMasker
from tqdm import tqdm
import nibabel as nib

#####################################################################
# Data loading
#####################################################################
data_dir = '/home/f_moldovan/projects/case_studies/data/bids'
layout = BIDSLayout(data_dir, derivatives=True)

# Get list of beta files for all subjects
subs = layout.get_subjects()
file_lists = []
for sub in subs:
    file_list = glob.glob(os.path.join(data_dir, 'derivatives', 'betas', f'sub-{sub}', 'beta_*'))
    file_list = [x for x in file_list]
    file_list = sorted(
        file_list,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0].replace("word", ""))
)
    file_lists.append(file_list)

# Extract condition names (word labels) from filenames of the first subject (assuming all subjects have the same conditions in the same order)
file_list = file_lists[0]  # Get the file list for the first subject
conditions = [os.path.basename(x).split("_")[1].split(".")[0] for x in file_list]

# Replace word labels with actual words used (and translate from Chinese to English)
alignment = pd.read_csv("/home/f_moldovan/projects/case_studies/data/bids/derivatives/annotations/align.csv")
chinese_words = [alignment[alignment['Con_Name'] == condition]['stimulus'].iloc[0] for condition in conditions]

translations = pd.read_csv("/home/f_moldovan/projects/case_studies/data/bids/derivatives/annotations/672words_translations.csv", header=None)
conditions_english = []
for chinese in chinese_words:
    translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
    conditions_english.append(translation) # builiding list of 672 English words corresponding to the 672 conditions
print("Sample conditions (English):", conditions_english[:10]) # same order as in conditions list, but now in English (word1 = dwarf, word2 = love, etc.)

# Load the mask 
mask_path = "/home/f_moldovan/projects/case_studies/data/brain_parcellations/emotion_parcellation_rsa_union.nii.gz"

#####################################################################
# Similarity matrix between all conditions/words for each ROI 
# for all subjects
#####################################################################
# Create a masker to extract voxel data in the same space as the mask
# Load parcel labels directly
mask_img = nib.load(mask_path)
mask_array = mask_img.get_fdata().astype(int).flatten()  # (91*109*91,) = full volume

# Flatten beta data the same way, but only keep voxels inside the parcellation
brain_mask = mask_array > 0  # (n_full_voxels,)
shared_masker = NiftiMasker(mask_img=mask_path)
shared_masker.fit()  # Fit the masker to the mask image
beta_data = shared_masker.transform(image.concat_imgs(file_list))  # (672, 84067)

# Get parcel labels for only the masked voxels
parcel_labels = mask_array[brain_mask]  # (84067,)

print("Parcel labels shape:", parcel_labels.shape)
print("Unique labels:", np.unique(parcel_labels))

similarity_matrices_subs = []
for sub in tqdm(subs, desc="Processing subjects"):
    file_list = file_lists[subs.index(sub)]  # Get the file list for the current subject
    beta_data = shared_masker.transform(image.concat_imgs(file_list))  # (672, 84067)

    similarity_matrices_rois = []
    for roi_id in np.unique(parcel_labels):
        if roi_id == 0:
            continue  # skip background

        roi_mask = parcel_labels == roi_id  # boolean index, same space as beta_data

        roi_data = beta_data[:, roi_mask]   # (n_conditions, n_roi_voxels)

        dist_matrix = pairwise_distances(roi_data, metric='correlation')
        similarity = 1 - dist_matrix
    
        adj = Adjacency(similarity, matrix_type='similarity', labels=conditions_english)
        similarity_matrices_rois.append(adj)

    # Store the list of similarity matrices for this subject
    similarity_matrices_subs.append(similarity_matrices_rois)

    # Save the similarity matrices for this subject to disk
    output_dir = os.path.join(data_dir, 'derivatives', 'similarity_matrices', f'sub-{sub}')
    os.makedirs(output_dir, exist_ok=True)
    # Convert Adjacency objects to plain 2D numpy arrays before saving.
    # nltools.Adjacency.data can be 1D (condensed) or 2D (square). Handle both.
    from scipy.spatial.distance import squareform
    arrays = []
    for adj in similarity_matrices_rois:
        data = getattr(adj, 'data', adj)
        arr = np.asarray(data)
        if arr.ndim == 1:
            # convert condensed vector to square matrix
            arr = squareform(arr)
        arrays.append(arr)

    # Save as a compressed NPZ with one array per ROI (keys: roi_0, roi_1, ...)
    save_dict = {f'roi_{i}': arrays[i] for i in range(len(arrays))}
    np.savez_compressed(os.path.join(output_dir, 'similarity_matrices.npz'), **save_dict)
print(f"Computed similarity matrices for {len(similarity_matrices_subs)} subjects, each with {len(similarity_matrices_subs[0])} ROI-specific matrices, meaning {len(similarity_matrices_subs) * len(similarity_matrices_subs[0])} total similarity matrices.")

#####################################################################
# Example: visualize the similarity matrix for the first ROI of the 
# first subject
#####################################################################
similarity_matrices = similarity_matrices_subs[0]  # Get the similarity matrices for the first subject

roi_id = 1  # Change this to visualize different ROIs
roi_array = (mask_array.reshape(91, 109, 91) == roi_id).astype(np.float32)
roi_nifti = nib.Nifti1Image(roi_array, mask_img.affine, mask_img.header)

plotting.plot_roi(roi_nifti, colorbar = False, title=f"ROI {roi_id} Mask")
save_path = '/home/f_moldovan/projects/case_studies/reports/figures/examples'
plt.savefig(os.path.join(save_path, f'sub-01_roi_{roi_id}_mask.png'), dpi=300)

matrix_idx = 0 if roi_id == 0 else roi_id - 1
similarity_matrices[matrix_idx].labels = conditions_english
similarity_matrices[matrix_idx].plot(vmin=-1, vmax=1, cmap='seismic')
save_path = '/home/f_moldovan/projects/case_studies/reports/plots/examples'
plt.savefig(os.path.join(save_path, f'sub-01_roi_{roi_id}_similarity.png'), dpi=300)

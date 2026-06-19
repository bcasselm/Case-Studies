'''
Computes the similarity matrices between the neural activation patterns (beta maps) for each of the 672 conditions/words, 
separately for each of the 39 ROIs in our Neurosynth meta-analytic parcellation, and separately for each subject. 
The similarity metric used is correlation distance (1 - Pearson correlation), which is commonly used in RSA studies. 
The resulting similarity matrices are stored as Adjacency objects from nltools, which can be easily manipulated and visualized later on.
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
# Configuration
#####################################################################
BASE_DIR = '/home/f_moldovan/projects/case_studies'
BIDS_DIR = os.path.join(BASE_DIR, 'data', 'bids')
ALIGNMENT_FILE_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', 'align.csv') # This file contains the mapping from condition names (e.g., word1, word2) to the actual Chinese words used in the experiment.
TRANSLATIONS_FILE_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', '672words_translations.csv') # This file contains the mapping from Chinese words to their English translations
MASK_PATH = os.path.join(BASE_DIR, 'data', 'brain_parcellations', 'emotion_parcellation_rsa_union.nii.gz') # This is the path to the parcellation mask we will use to extract voxel data for each ROI. It should be in the same space as the beta maps (e.g., MNI space).
OUT_DIR = os.path.join(BIDS_DIR, 'derivatives', 'similarity_matrices') # Directory where we will save the computed similarity matrices for each subject and ROI.
FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures', 'examples') # Directory to save example figures of ROI masks.
PLOT_DIR = os.path.join(BASE_DIR, 'reports', 'plots', 'examples') # Directory to save example plots of similarity matrices.
EXAMPLE_ROI = 28 # We will visualize the similarity matrix and actual mask for this ROI as an example. You can change this to visualize different ROIs.

#####################################################################
# Helper functions
#####################################################################
def load_beta_files(bids_dir):
    layout = BIDSLayout(bids_dir, derivatives=True)

    # Get list of beta files for all subjects
    subs = layout.get_subjects()
    file_lists = []
    for sub in subs:
        file_list = glob.glob(os.path.join(bids_dir, 'derivatives', 'betas', f'sub-{sub}', 'beta_*'))
        file_list = [x for x in file_list]
        file_list = sorted(
            file_list,
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0].replace("word", ""))
    )
        file_lists.append(file_list)
        
    return subs, file_lists

def get_english_conditions(file_list, alignment_path, translations_path):
    # Extract condition names (word labels) from filenames of the first subject (assuming all subjects have the same conditions in the same order)
    conditions = [os.path.basename(x).split("_")[1].split(".")[0] for x in file_list]

    # Replace word labels with actual words used (and translate from Chinese to English)
    alignment = pd.read_csv(alignment_path)
    chinese_words = [alignment[alignment['Con_Name'] == condition]['stimulus'].iloc[0] for condition in conditions]

    translations = pd.read_csv(translations_path, header=None)
    conditions_english = []
    for chinese in chinese_words:
        translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
        conditions_english.append(translation) # builiding list of 672 English words corresponding to the 672 conditions
    print("Sample conditions (English):", conditions_english[:10]) # same order as in conditions list, but now in English (word1 = dwarf, word2 = love, etc.)
    
    return conditions_english

def compute_and_save_similarities(subs, file_lists, conditions_english, mask_path, out_dir):
    # Create a masker to extract voxel data in the same space as the mask
    # Load parcel labels directly
    mask_img = nib.load(mask_path)
    mask_array = mask_img.get_fdata().astype(int).flatten()  # (91*109*91,) = full volume

    # Flatten beta data the same way, but only keep voxels inside the parcellation
    brain_mask = mask_array > 0  # (n_full_voxels,)
    shared_masker = NiftiMasker(mask_img=mask_path)
    shared_masker.fit()  # Fit the masker to the mask image

    # Get parcel labels for only the masked voxels
    parcel_labels = mask_array[brain_mask] 

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
        output_dir = os.path.join(out_dir, f'sub-{sub}')
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

        # Save as a compressed NPZ with one array per ROI (keys: roi_1, roi_2, ...)
        # The original ROI IDs start from 1, so we'll use those for the keys.
        roi_ids = np.unique(parcel_labels)
        roi_ids = roi_ids[roi_ids != 0]  # Exclude background
        save_dict = {f'roi_{roi_ids[i]}': arrays[i] for i in range(len(arrays))}
        np.savez_compressed(os.path.join(output_dir, 'similarity_matrices.npz'), **save_dict)
    
    print(f"Computed similarity matrices for {len(similarity_matrices_subs)} subjects, each with {len(similarity_matrices_subs[0])} ROI-specific matrices, meaning {len(similarity_matrices_subs) * len(similarity_matrices_subs[0])} total similarity matrices.")
    
    return similarity_matrices_subs, mask_img, mask_array

def visualize_example_roi(similarity_matrices_subs, example_roi, mask_img, mask_array, conditions_english, fig_dir, plot_dir):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    similarity_matrices = similarity_matrices_subs[0]  # Get the similarity matrices for the first subject

    roi_array = (mask_array.reshape(91, 109, 91) == example_roi).astype(np.float32)
    roi_nifti = nib.Nifti1Image(roi_array, mask_img.affine, mask_img.header)

    plotting.plot_roi(roi_nifti, colorbar = False, title=f"ROI {example_roi} Mask", draw_cross=False, black_bg = True)
    plt.savefig(os.path.join(fig_dir, f'sub-01_roi_{example_roi}_mask.png'), dpi=300)
    matrix_idx = 0 if example_roi == 0 else example_roi - 1
    similarity_matrices[matrix_idx].labels = conditions_english
    similarity_matrices[matrix_idx].plot(vmin=-1, vmax=1, cmap='seismic')
    plt.savefig(os.path.join(plot_dir, f'sub-01_roi_{example_roi}_similarity.png'), dpi=300)


#####################################################################
# Main execution
#####################################################################
if __name__ == "__main__":
    # 1. Data loading
    subs, file_lists = load_beta_files(BIDS_DIR)
    
    # 2. Assign and translate conditions
    conditions_english = get_english_conditions(
        file_lists[0], ALIGNMENT_FILE_PATH, TRANSLATIONS_FILE_PATH
    )
    
    # 3. Similarity matrix between all conditions/words for each ROI for all subjects
    similarity_matrices_subs, mask_img, mask_array = compute_and_save_similarities(
        subs, file_lists, conditions_english, MASK_PATH, OUT_DIR
    )
    
    # 4. Example: visualize the similarity matrix for the EXAMPLE_ROI of the first subject
    visualize_example_roi(
        similarity_matrices_subs, EXAMPLE_ROI, mask_img, mask_array, 
        conditions_english, FIG_DIR, PLOT_DIR
    )
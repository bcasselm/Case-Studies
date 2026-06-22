'''
Computes the similarity matrices between the neural activation patterns (beta maps) for each word/condition,
separately for each ROI in our Neurosynth meta-analytic parcellation, and separately for each subject.
The similarity metric is correlation distance (1 - Pearson correlation), which is commonly used in RSA studies.
The resulting similarity matrices are stored as Adjacency objects from nltools, which can be easily manipulated and visualized later on.
'''

#####################################################################
# Imports
#####################################################################
import os
import glob
from pathlib import Path
from nilearn import image, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltools.data import Adjacency
from sklearn.metrics import pairwise_distances
from nilearn.maskers import NiftiMasker
from scipy.spatial.distance import squareform
from tqdm import tqdm
import nibabel as nib

#####################################################################
# Configuration
#####################################################################
PROJECT_ROOT = Path("/Volumes/T9/Birgit")

DATA_DIR = PROJECT_ROOT / "data"
BIDS_DIR = DATA_DIR / "ds004301"
ANNOTATIONS_DIR = BIDS_DIR / "derivatives" / "annotations"
ALIGNMENT_FILE_PATH = ANNOTATIONS_DIR / "align.csv"                                     # maps condition names (word1, word2, ...) to the Chinese stimulus words
TRANSLATIONS_FILE_PATH = ANNOTATIONS_DIR / "672words_translations.csv"                  # maps Chinese words to their English translations
MASK_PATH = DATA_DIR / "brain_parcellations" / "emotion_parcellation_rsa_union.nii.gz"  # parcellation mask (same space as the beta maps)
BETAS_DIR = BIDS_DIR / "derivatives" / "betas"                                          # per-word beta maps written by script 02
OUT_DIR = BIDS_DIR / "derivatives" / "similarity_matrices"                              # where the similarity matrices are saved
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "examples"                             # example ROI masks
PLOT_DIR = PROJECT_ROOT / "reports" / "plots" / "examples"                              # example similarity matrices
EXAMPLE_ROI = 20                                                                        # ROI to visualize as an example

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SUBJECTS = [f"{i:02d}" for i in range(1, 12)]

#####################################################################
# Data loading
#####################################################################
# Get list of beta files for all subjects, sorted by word number
file_lists = []
for sub in SUBJECTS:
    file_list = glob.glob(os.path.join(BETAS_DIR, f'sub-{sub}', 'beta_*'))
    file_list = sorted(
        file_list,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0].replace("word", ""))
    )
    file_lists.append(file_list)

# Extract condition names (word labels) from filenames of the first subject
file_list = file_lists[0]
conditions = [os.path.basename(x).split("_")[1].split(".")[0] for x in file_list]

# Replace word labels with the actual words used, then translate from Chinese to English
alignment = pd.read_csv(ALIGNMENT_FILE_PATH, encoding="utf-8-sig")
chinese_words = [alignment[alignment['Con_Name'] == condition]['stimulus'].iloc[0] for condition in conditions]

translations = pd.read_csv(TRANSLATIONS_FILE_PATH, header=None, encoding="utf-8-sig")
conditions_english = []
for chinese in chinese_words:
    translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
    conditions_english.append(translation)
print("Sample conditions (English):", conditions_english[:10])

#####################################################################
# Similarity matrix between all conditions/words for each ROI
# for all subjects
#####################################################################
# Resample the parcellation mask to the native beta grid (nearest-neighbour so
# integer ROI labels are preserved).  The betas live at ~3 mm; the raw mask is
# 2 mm.  Using the raw mask would make NiftiMasker upsample every beta to 2 mm,
# which interpolates the data and undermines the multivariate patterns.
_ref_beta = nib.load(file_lists[0][0])
mask_img = image.resample_to_img(str(MASK_PATH), _ref_beta, interpolation="nearest")
del _ref_beta

mask_array = mask_img.get_fdata().astype(int).flatten()
brain_mask = mask_array > 0

shared_masker = NiftiMasker(mask_img=mask_img)
shared_masker.fit()

# Parcel labels for the masked voxels (same order as the masker output)
parcel_labels = mask_array[brain_mask]
print("Parcel labels shape:", parcel_labels.shape)
print("Unique labels:", np.unique(parcel_labels))
print("NiftiMasker n_voxels:", shared_masker.mask_img_.get_fdata().astype(bool).sum())

# Per-ROI voxel counts at the native beta resolution
roi_ids_check = np.unique(parcel_labels)
roi_ids_check = roi_ids_check[roi_ids_check != 0]
print("Voxels per ROI (native grid):",
      {int(r): int((parcel_labels == r).sum()) for r in roi_ids_check})

similarity_matrices_subs = []
for sub in tqdm(SUBJECTS, desc="Processing subjects"):
    file_list = file_lists[SUBJECTS.index(sub)]
    beta_data = shared_masker.transform(image.concat_imgs(file_list))  # (n_conditions, n_in_mask_voxels)

    similarity_matrices_rois = []
    for roi_id in np.unique(parcel_labels):
        if roi_id == 0:
            continue  # skip background

        roi_mask = parcel_labels == roi_id        
        roi_data = beta_data[:, roi_mask]         

        dist_matrix = pairwise_distances(roi_data, metric='correlation')
        similarity = 1 - dist_matrix

        adj = Adjacency(similarity, matrix_type='similarity', labels=conditions_english)
        similarity_matrices_rois.append(adj)

    # Store the list of similarity matrices for this subject
    similarity_matrices_subs.append(similarity_matrices_rois)

    # Save the similarity matrices for this subject to disk
    output_dir = os.path.join(OUT_DIR, f'sub-{sub}')
    os.makedirs(output_dir, exist_ok=True)
    # Convert Adjacency objects to plain 2D numpy arrays before saving
    arrays = []
    for adj in similarity_matrices_rois:
        data = getattr(adj, 'data', adj)
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = squareform(arr)
        arrays.append(arr)

    # Save as a compressed NPZ with one array per ROI (keys: roi_1, roi_2, ...)
    roi_ids = np.unique(parcel_labels)
    roi_ids = roi_ids[roi_ids != 0]  # exclude background
    save_dict = {f'roi_{roi_ids[i]}': arrays[i] for i in range(len(arrays))}
    np.savez_compressed(os.path.join(output_dir, 'similarity_matrices.npz'), **save_dict)
print(f"Computed similarity matrices for {len(similarity_matrices_subs)} subjects, each with {len(similarity_matrices_subs[0])} ROI-specific matrices, meaning {len(similarity_matrices_subs) * len(similarity_matrices_subs[0])} total similarity matrices.")

#####################################################################
# Example: visualize the similarity matrix and mask for one ROI
# of the first subject
#####################################################################
similarity_matrices = similarity_matrices_subs[0]

roi_array = (mask_array.reshape(mask_img.shape[:3]) == EXAMPLE_ROI).astype(np.float32)
roi_nifti = nib.Nifti1Image(roi_array, mask_img.affine)

plotting.plot_roi(roi_nifti, colorbar=False, title=f"ROI {EXAMPLE_ROI} Mask", draw_cross=False, black_bg=True)
plt.savefig(os.path.join(FIG_DIR, f'sub-01_roi_{EXAMPLE_ROI}_mask.png'), dpi=300)
matrix_idx = 0 if EXAMPLE_ROI == 0 else EXAMPLE_ROI - 1
similarity_matrices[matrix_idx].labels = conditions_english
similarity_matrices[matrix_idx].plot(vmin=-1, vmax=1, cmap='seismic')
plt.savefig(os.path.join(PLOT_DIR, f'sub-01_roi_{EXAMPLE_ROI}_similarity.png'), dpi=300)

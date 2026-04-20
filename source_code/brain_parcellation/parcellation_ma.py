'''
Meta-Analysis for Emotional Concept/Word Representation Brain Parcellation

This script performs a meta-analysis to identify brain regions associated with the concept of "emotional concept representation" using 
the Neurosynth database. It correlates topic-based features with specific 
terms related to emotion and semantics to find relevant topics. 
'''

#####################################################################
# Imports
#####################################################################
import nibabel as nib
import numpy as np
from nilearn import image, plotting
from nimare.extract import fetch_neurosynth
from nimare.meta.cbma import mkda
import numpy as np
from scipy import ndimage
from nimare.correct import FDRCorrector
from pathlib import Path

#####################################################################
# Helper functions
#####################################################################
def print_mask_space_info(mask_img, mask_path):
    header = mask_img.header
    print(f"Space info for {mask_path}:")
    print(f"  shape: {mask_img.shape}")
    print(f"  zooms: {header.get_zooms()[:3]}")
    print(f"  qform_code: {int(header['qform_code'])}")
    print(f"  sform_code: {int(header['sform_code'])}")
    print(f"  affine:\n{mask_img.affine}")

#####################################################################
# Load the dataset (dataset object)
#####################################################################
ns_data = fetch_neurosynth(data_dir = "/home/f_moldovan",
                           version= "7", 
                           source= "abstract", 
                           vocab= "LDA400",
                           target = "mni152_2mm",
                           return_type= "dataset")[0]

#####################################################################
# Get the correlation between 'LDA400_X' and our target concepts
#####################################################################
# We look for topics that relate to BOTH 'emotion' and 'semantic'/'word'
features = ns_data.annotations.columns
print("Available feature columns:")
print(features[:20])  # Print the first 20 columns to check the format

topic_cols = [c for c in features if c.startswith('LDA400_')]

# Identify our target word columns (the actual words from the abstracts)
target_terms = ["emotion", "semantic", "word", "concept"]
matched_term_cols = []
for term in target_terms:
    matches = [col for col in topic_cols if term in col.lower()]
    if matches:
        matched_term_cols.extend(matches)

matched_term_cols = sorted(set(matched_term_cols))
if not matched_term_cols:
    raise ValueError(
        "No topic columns matched target terms. "
        "Try a different vocabulary (e.g., vocab='terms') or target terms."
    )

print("Matched term-related columns:")
print(matched_term_cols[:10])

term_weights = ns_data.annotations[matched_term_cols]

# Calculate which topics correlate most with our conceptual-emotion cluster
topic_corrs = ns_data.annotations[topic_cols].corrwith(term_weights.mean(axis=1))

# Sort to find the winner
top_topics = topic_corrs.sort_values(ascending=False)
print("Top Topics for 'Emotional Concept Representation':")
print(top_topics.head(10))

#####################################################################
# Use top topics to perform meta-analysis and identify brain regions
#####################################################################
# Create meta-analytic maps for the top topics and identify parcels for RSA
for topic in top_topics.head(7).index: # top 7 because below 7 have corr < 0.2
    topic_weights = ns_data.annotations[topic]
    threshold = topic_weights.quantile(0.70)  # Keep top 30% of studies for this topic
    study_ids = ns_data.annotations[ns_data.annotations[topic] > threshold]["id"].tolist()

    # Run MKDA on those specific studies
    # First, split the preselected dataset into 2
    ns_data_1 = ns_data.slice(study_ids[:len(study_ids) // 2])
    ns_data_2 = ns_data.slice(study_ids[len(study_ids) // 2:])

    mkda_estimator = mkda.MKDAChi2()
    results = mkda_estimator.fit(ns_data_1, ns_data_2)

    # Perform cluster-level FDR correction on the results
    fdr_corrector = FDRCorrector(method="indep", alpha=0.05)
    results_fdr = fdr_corrector.transform(results)

    # Get corrected maps: threshold by corrected p-values, keep positive effects.
    z_map_name = None
    for key in results_fdr.maps.keys():
        if "z_desc-association" in key:
            z_map_name = key
            break
    if z_map_name is None:
        raise ValueError(f"No association z-map found. Available maps: {list(results_fdr.maps.keys())}")

    p_corr_map_name = None
    for key in results_fdr.maps.keys():
        key_lower = key.lower()
        if "p" in key_lower and ("corr-fdr" in key_lower or "_corr-" in key_lower):
            p_corr_map_name = key
            break
    if p_corr_map_name is None:
        raise ValueError(f"No corrected p-map found. Available maps: {list(results_fdr.maps.keys())}")

    z_map = results_fdr.get_map(z_map_name)
    p_corr_map = results_fdr.get_map(p_corr_map_name)

    z_data = np.nan_to_num(z_map.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
    p_corr_data = np.nan_to_num(p_corr_map.get_fdata(), nan=1.0, posinf=1.0, neginf=1.0)

    significant_positive = (p_corr_data < 0.05) & (z_data > 0)
    thresholded_data = np.where(significant_positive, z_data, 0.0)
    thresholded_map = image.new_img_like(z_map, thresholded_data)

    # Label connected components on the thresholded binary mask
    binary_mask_img = image.math_img("img > 0", img=thresholded_map)
    binary_mask = binary_mask_img.get_fdata().astype(bool)
    labeled_map, n_labels = ndimage.label(binary_mask)

    # Drop tiny components (min_size in voxels), then relabel sequentially
    min_size = 100
    cleaned_map = np.zeros_like(labeled_map, dtype=np.int32)
    new_label = 1
    for label_idx in range(1, n_labels + 1):
        component = labeled_map == label_idx
        if int(component.sum()) >= min_size:
            cleaned_map[component] = new_label
            new_label += 1

    n_parcels = new_label - 1
    if n_parcels == 0:
        print(f"No parcels survived thresholding for topic: {topic}")
        continue

    parcel_img = image.new_img_like(thresholded_map, cleaned_map)

    # visualize & export
    plotting.plot_roi(parcel_img, title="Meta-Analytic Emotion Parcels")
    output_path = f"data/brain_parcellations/emotion_parcellation_rsa_{topic}.nii.gz"
    parcel_img.to_filename(output_path)
    print_mask_space_info(parcel_img, output_path) # all in MNI152 2mm space

    print(f"Created {n_parcels} parcels for RSA for topic: {topic}")

#####################################################################
# Merge all topic-specific parcels into a single parcellation for RSA
#####################################################################
# Preserve each original parcel identity across topic-specific masks.
# Each input parcel gets its own global label in the final map.

mask_paths = []
for topic in top_topics.head(7).index:
    p = Path(f"data/brain_parcellations/emotion_parcellation_rsa_{topic}.nii.gz")
    if p.exists():
        mask_paths.append(p)

if not mask_paths:
    raise ValueError("No topic-specific masks found to merge.")

# Use first mask as reference grid/affine
ref_img = image.load_img(str(mask_paths[0]))
final_label_map = np.zeros(ref_img.shape, dtype=np.int32)
next_global_label = 1
min_size_union = 100
skipped_small_parcels = 0
overlap_voxel_count = 0

# Keep every original parcel as a unique global label.
# If parcels overlap in voxel space, first-assigned label is kept.
for p in mask_paths:
    img = image.load_img(str(p))
    data = np.nan_to_num(img.get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
    local_labels = [int(label) for label in np.unique(data) if label > 0]

    for local_label in local_labels:
        local_parcel = data == local_label
        local_size = int(local_parcel.sum())

        if local_size < min_size_union:
            skipped_small_parcels += 1
            continue

        free_voxels = local_parcel & (final_label_map == 0)
        free_size = int(free_voxels.sum())

        # If all voxels overlap with previously assigned parcels, record overlap and skip
        if free_size == 0:
            overlap_voxel_count += local_size
            continue

        # Require that the remaining free portion of the parcel still meets the
        # min-size threshold. Otherwise skip it to avoid creating very small
        # fragmented parcels in the union.
        if free_size < min_size_union:
            skipped_small_parcels += 1
            overlap_voxel_count += (local_size - free_size)
            continue

        # Assign the free voxels to a new global label
        overlap_voxel_count += (local_size - free_size)
        final_label_map[free_voxels] = next_global_label
        next_global_label += 1

n_union_parcels = next_global_label - 1
if n_union_parcels == 0:
    raise ValueError("No parcels survived filtering in the union mask.")

final_parcellation = image.new_img_like(ref_img, final_label_map)
final_path = "data/brain_parcellations/emotion_parcellation_rsa_union.nii.gz"

# Relabel to contiguous 1..N to guarantee downstream tools see compact labels
unique_labels = np.unique(final_label_map)
unique_labels = unique_labels[unique_labels != 0] # Exclude background (0) from relabeling
if unique_labels.size > 0:
    mapping = {int(old): int(new) for new, old in enumerate(sorted(unique_labels), start=1)}
    relabeled_map = np.zeros_like(final_label_map, dtype=np.int32)
    for old_label, new_label in mapping.items():
        relabeled_map[final_label_map == old_label] = new_label
    final_parcellation = image.new_img_like(ref_img, relabeled_map)
    n_union_parcels = len(mapping)
    print(f"Relabeled union map to contiguous labels 1..{n_union_parcels}")
    print("Sample label mapping (old->new):", dict(list(mapping.items())[:10]))
else:
    # no parcels found (should not happen here)
    mapping = {}

final_parcellation.to_filename(final_path)
print_mask_space_info(final_parcellation, final_path)
print(f"Saved identity-preserving union parcellation with {n_union_parcels} parcels: {final_path}")
print(f"Skipped small parcels (< {min_size_union} voxels): {skipped_small_parcels}")
print(f"Overlapping voxels assigned by first-come rule: {overlap_voxel_count}")

#####################################################################
# Sanity checks
#####################################################################
img = nib.load("data/brain_parcellations/emotion_parcellation_rsa_union.nii.gz")
data = img.get_fdata().astype(int)
print("Background present?", 0 in np.unique(data))
print("Parcel labels (sample):", [v for v in np.unique(data)[:10]])
print("Number of parcels (excluding background):", len(np.unique(data)) - (1 if 0 in np.unique(data) else 0))
print("Parcel sizes (sample):", {label: int((data == label).sum()) for label in np.unique(data) if label != 0})

#####################################################################
# Visualize union mask and save to disk
#####################################################################
fig = plotting.plot_roi(final_parcellation, title="Union Parcellation for RSA",
                        display_mode="mosaic", colorbar=True)
fig_path = "reports/figures/brain_parcellation/emotion_parcellation_rsa_union_plot.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved union parcellation figure to: {fig_path}")
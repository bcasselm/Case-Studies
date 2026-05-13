'''
Performs the RSA analysis between the neural similarity matrices and the BERT similarity matrix. 
This script is meant to be run after both the neural similarity matrices and the BERT similarity matrix have been computed 
and saved to disk by their respective scripts.

The script loads the neural similarity matrices for each subject and each ROI, as well as the BERT similarity matrix,
and then computes the Spearman correlation between the neural similarity matrices and the BERT similarity matrix for each subject and each ROI.

The resulting RSA correlation values are stored in a pandas DataFrame, which can be easily manipulated and visualized later on.
'''

#######################################################################
# Imports
#######################################################################
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from bids import BIDSLayout
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests

#######################################################################
# Data loading
#######################################################################
data_dir = '/home/f_moldovan/projects/case_studies/data/bids'
layout = BIDSLayout(data_dir, derivatives=True)

# Get list of neural similarity matrices for all subjects and all ROIs
# Assuming ROI-specific neural similarity matrices are stored into 1 file per subject, with one matrix per ROI (i.e., in a dictionary format)
subs = layout.get_subjects()
print("Subjects found in BIDS dataset:", subs)
neural_sim_files = []
for sub in tqdm(subs, desc="Finding neural similarity matrix files"):
    file_list = glob.glob(os.path.join(data_dir, 'derivatives', 'similarity_matrices', f'sub-{sub}', 'similarity_matrices.npz'))
    if len(file_list) == 0:
        print(f"Warning: No neural similarity matrix found for subject {sub}")
    else:
        neural_sim_files.append(file_list[0])  # Assuming there's only one file per subject
print("Neural similarity matrix files found:", len(neural_sim_files))

# Each file should contain a dictionary with ROI names as keys and similarity matrices as values
print("Sample neural similarity matrix file contents:")
for file in neural_sim_files[:3]:  # Print contents of first 3 files
    data = np.load(file, allow_pickle=True)
    print(f"File: {file}")
    for roi, matrix in data.items():
        print(f"  ROI: {roi}, Shape: {matrix.shape}")

# Load the BERT similarity matrix
BERT_sim_file = os.path.join(data_dir, 'derivatives', 'annotations', 'embeddings', 'contextual word embeddings', 'BERT_similarity_matrix_adj.npz')
if not os.path.exists(BERT_sim_file):
    raise FileNotFoundError(f"BERT similarity matrix file not found at {BERT_sim_file}. Please run the BERT_similarity.py script first to compute and save the BERT similarity matrix.")
bert_matrix = np.load(BERT_sim_file, allow_pickle=True)
print("BERT similarity matrix file contents:", bert_matrix.files)
bert_similarity_matrix_flat = bert_matrix['data']
bert_labels = bert_matrix['labels']
print("BERT similarity matrix shape:", bert_similarity_matrix_flat.shape) # need to reconstruct the matrix from the flattened data
print("BERT similarity matrix labels shape:", bert_labels.shape) # should be (672,) with the English word labels corresponding to the 672 conditions

# Reconstruct the 2D square matrix from the flattened 1D vector
bert_similarity_matrix = distance.squareform(bert_similarity_matrix_flat)
print("BERT similarity matrix shape:", bert_similarity_matrix.shape)

##########################################################################
# Compute RSA correlations between neural similarity matrices and BERT similarity matrix
# for each subject and each ROI
##########################################################################
rsa_results = []
for file in tqdm(neural_sim_files, desc="Computing RSA correlations"):
    # Extract subject ID from the file path
    # e.g., from '/.../sub-01/similarity_matrices.npz', get '01'
    subject_id = os.path.basename(os.path.dirname(file)).replace('sub-', '')

    data = np.load(file, allow_pickle=True)

    for roi, neural_matrix in data.items():
        # Compute Spearman correlation between the neural similarity matrix and the BERT similarity matrix
        # We need to flatten both matrices to compute the correlation, but only the upper triangle (excluding the diagonal) to avoid redundancy
        # Get the upper triangle indices
        triu_indices = np.triu_indices_from(neural_matrix, k=1)
        neural_values = neural_matrix[triu_indices]
        bert_values = bert_similarity_matrix[triu_indices]
        rsa_corr, _ = spearmanr(neural_values, bert_values)
        rsa_results.append({'rsa_corr': rsa_corr, 'subject': subject_id, 'roi': roi})
# Convert results to a pandas DataFrame
rsa_df = pd.DataFrame(rsa_results)
print("RSA results DataFrame:")
print(rsa_df.head())

########################################################################
# Visualize the RSA results using a boxplot to show the distribution of 
# RSA correlations across subjects for each ROI
########################################################################
plt.figure(figsize=(12, 15))
sns.violinplot(x='rsa_corr', y='roi', data=rsa_df, hue = 'roi', inner='box')
plt.title('RSA Correlation (Spearman) Between Neural Similarity And BERT Similarity Across ROIs')
plt.xlabel('Spearman Correlation (RSA)')
plt.ylabel('ROI')
plt.xticks(rotation=45)
plt.vlines(x=0, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color='grey', linestyle='--')  # Add a vertical line at 0
plt.tight_layout()
plt.savefig('/home/f_moldovan/projects/case_studies/reports/plots/rsa_results/rsa_violinplot.png', dpi=300)

# Also visualize distribution of RSA correlations across ROIs for each subject
plt.figure(figsize=(12, 6))
sns.violinplot(x='rsa_corr', y='subject', data=rsa_df, hue = 'subject', inner='box')
plt.title('RSA Correlation (Spearman) Between Neural Similarity And BERT Similarity Across Subjects')
plt.xlabel('Spearman Correlation (RSA)')
plt.ylabel('Subject')
plt.xticks(rotation=45)
plt.vlines(x=0, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color='grey', linestyle='--')  # Add a vertical line at 0
plt.tight_layout()
plt.savefig('/home/f_moldovan/projects/case_studies/reports/plots/rsa_results/rsa_violinplot_subjects.png', dpi=300)

#######################################################################
# Test for one-sided significance of RSA correlations across subjects for each ROI
# Nonparametric tests and robust central tendency measures are used due to the 
# small sample size (n=11 subjects) and potential non-normality of the RSA correlation distributions.
#######################################################################
# First, compute median RSA correlation across subjects for each ROI
median_rsa_by_roi = rsa_df.groupby('roi')['rsa_corr'].median()
print("Median RSA correlation by ROI:")
print(median_rsa_by_roi)

# Then, compute 95% confidence intervals for the median RSA correlation across subjects for each ROI using bootstrapping
bootstrap_results = []
for roi in rsa_df['roi'].unique():
    roi_data = rsa_df[rsa_df['roi'] == roi]['rsa_corr'].values
    # Perform bootstrapping
    n_bootstraps = 10000
    bootstrapped_medians = []
    for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping ROI {roi}"):
        boot_sample = np.random.choice(roi_data, size=len(roi_data), replace=True)
        boot_median = np.median(boot_sample)
        bootstrapped_medians.append(boot_median)
    # Compute 95% confidence interval
    lower_ci = np.percentile(bootstrapped_medians, 2.5)
    upper_ci = np.percentile(bootstrapped_medians, 97.5)
    bootstrap_results.append({'roi': roi, 'median_rsa': median_rsa_by_roi[roi], 'lower_ci': lower_ci, 'upper_ci': upper_ci})
bootstrap_df = pd.DataFrame(bootstrap_results)
print("Bootstrap results with confidence intervals for median RSA correlation by ROI:")
print(bootstrap_df)

# Finally, perform one-sided (greater than 0) permutation test for each ROI to see if the median RSA correlation is significantly greater than 0
permutation_results = []
for roi in rsa_df['roi'].unique():
    roi_data = rsa_df[rsa_df['roi'] == roi]['rsa_corr'].values
    observed_median = median_rsa_by_roi[roi]
    # Permutation test
    n_permutations = 10000
    permuted_medians = []
    n_subjects = len(roi_data)
    for _ in tqdm(range(n_permutations), desc=f"Permutation testing ROI {roi}"):
        # To test against the null hypothesis of zero median, we randomly flip the signs of the data
        random_signs = np.random.choice([-1, 1], size=n_subjects, replace=True)
        permuted_data = roi_data * random_signs
        permuted_median = np.median(permuted_data)
        permuted_medians.append(permuted_median)
    # Compute p-value as the proportion of permuted medians (from the null distribution)
    # that are greater than or equal to the observed median.
    p_value = np.mean(np.array(permuted_medians) >= observed_median)
    permutation_results.append({'roi': roi, 'observed_median': observed_median, 'p_value': p_value})
permutation_df = pd.DataFrame(permutation_results)
print("Permutation test results for median RSA correlation by ROI:")

# Correct for multiple comparisons across ROIs using False Discovery Rate (FDR)
p_values = permutation_df['p_value'].values
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
permutation_df['p_value_corrected'] = pvals_corrected
permutation_df['significant'] = reject
print("Permutation test results with FDR correction:")
print(permutation_df)

#########################################################################
# Visualize the permutation distribution of median RSA correlations for each ROI,
# with the observed median and confidence intervals
#########################################################################
for roi in rsa_df['roi'].unique():
    roi_data = rsa_df[rsa_df['roi'] == roi]['rsa_corr'].values
    observed_median = median_rsa_by_roi[roi]
    lower_ci = bootstrap_df[bootstrap_df['roi'] == roi]['lower_ci'].values[0]
    upper_ci = bootstrap_df[bootstrap_df['roi'] == roi]['upper_ci'].values[0]

    # Plot the distribution of permuted medians
    plt.figure(figsize=(8, 6))
    sns.histplot(permuted_medians, bins=30, kde=True)
    plt.axvline(observed_median, color='red', linestyle='--', label='Observed Median')
    plt.axvline(lower_ci, color='blue', linestyle='--', label='95% CI Lower')
    plt.axvline(upper_ci, color='blue', linestyle='--', label='95% CI Upper')
    plt.title(f'Permutation Distribution of Median RSA Correlations for ROI {roi}')
    plt.xlabel('Median RSA Correlation (Permuted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/home/f_moldovan/projects/case_studies/reports/plots/rsa_results/permutation_distribution_{roi}.png', dpi=300)
    plt.close()

#########################################################################
# Now repeat everything above but now also across ROIs (still across subjects) 
# to see if the median RSA correlation across all ROIs is significantly greater than 0
#########################################################################
# Compute median RSA correlation across all ROIs and all subjects
median_rsa_all = rsa_df['rsa_corr'].median()
print("Median RSA correlation across all ROIs and subjects:", median_rsa_all)

# Compute 95% confidence interval for the median RSA correlation across all ROIs and subjects using bootstrapping
n_bootstraps = 10000
bootstrapped_medians_all = []
for _ in tqdm(range(n_bootstraps), desc="Bootstrapping across all ROIs and subjects"):
    boot_sample_all = np.random.choice(rsa_df['rsa_corr'].values, size=len(rsa_df), replace=True)
    boot_median_all = np.median(boot_sample_all)
    bootstrapped_medians_all.append(boot_median_all)
lower_ci_all = np.percentile(bootstrapped_medians_all, 2.5)
upper_ci_all = np.percentile(bootstrapped_medians_all, 97.5)
print("95% confidence interval for median RSA correlation across all ROIs and subjects:", (lower_ci_all, upper_ci_all))
bootstrap_df_all = pd.DataFrame({'bootstrapped_median': bootstrapped_medians_all, 'lower_ci': lower_ci_all, 'upper_ci': upper_ci_all})

# Perform one-sided (greater than 0) permutation test for the median RSA correlation across all ROIs and subjects
observed_median_all = median_rsa_all
n_permutations = 10000
permuted_medians_all = []
n_samples_all = len(rsa_df)
for _ in tqdm(range(n_permutations), desc="Permutation testing across all ROIs and subjects"):
    random_signs_all = np.random.choice([-1, 1], size=n_samples_all, replace=True)
    permuted_data_all = rsa_df['rsa_corr'].values * random_signs_all
    permuted_median_all = np.median(permuted_data_all)
    permuted_medians_all.append(permuted_median_all)
p_value_all = np.mean(np.array(permuted_medians_all) >= observed_median_all)
print("Permutation test p-value for median RSA correlation across all ROIs and subjects:", p_value_all)
permutation_df_all = pd.DataFrame({'observed_median': observed_median_all, 'p_value': p_value_all})

# Visualize the permutation distribution of median RSA correlations with bootstrapped CI (across all ROIs and subjects)
plt.figure(figsize=(8, 6))
sns.histplot(permuted_medians_all, bins=30, kde=True)
plt.axvline(observed_median_all, color='red', linestyle='--', label='Observed Median')
plt.axvline(lower_ci_all, color='blue', linestyle='--', label='95% CI Lower')
plt.axvline(upper_ci_all, color='blue', linestyle='--', label='95% CI Upper')
plt.title('Permutation Distribution of Median RSA Correlations across all ROIs and subjects')
plt.xlabel('Median RSA Correlation (Permuted)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('/home/f_moldovan/projects/case_studies/reports/plots/rsa_results/permutation_distribution_all_rois.png', dpi=300)
plt.close()

########################################################################
# Save the RSA results DataFrame to disk for later use
########################################################################
rsa_df.to_csv('/home/f_moldovan/projects/case_studies/data/rsa_results/rsa_corrs.csv', index=False)
bootstrap_df.to_csv('/home/f_moldovan/projects/case_studies/data/rsa_results/bootstrap_results.csv', index=False)
permutation_df.to_csv('/home/f_moldovan/projects/case_studies/data/rsa_results/permutation_results.csv', index=False)

bootstrap_df_all.to_csv('/home/f_moldovan/projects/case_studies/data/rsa_results/bootstrap_results_all_rois.csv', index=False)
permutation_df_all.to_csv('/home/f_moldovan/projects/case_studies/data/rsa_results/permutation_results_all_rois.csv', index=False)
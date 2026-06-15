'''
Performs a dual emotion-modulated RSA analysis to test whether the alignment between LLM and neural similarity is modulated by
positivity or negativity ratings of words. This script includes:
1. Loading the LLM similarity matrix and the emotion ratings for the 672 words.
2. Creating continuous pairwise intensity matrices for positivity and negativity based on the ratings of the individual words.
3. For each subject and each ROI, fitting a quantile regression model where the dependent variable is the neural similarity and the independent variables are:
   - The LLM similarity (main effect)
   - The pairwise positivity intensity (main effect)
   - The pairwise negativity intensity (main effect)
   - The interaction between LLM similarity and pairwise positivity (positivity bias)
   - The interaction between LLM similarity and pairwise negativity (negativity bias)
4. Extracting the beta coefficients for the interaction terms to assess positivity and negativity bias in each ROI and subject.
5. Performing group-level statistical tests (Wilcoxon signed-rank test) on the interaction betas across subjects for each ROI,
and applying FDR correction for multiple comparisons.
6. Visualizing the results in a bar plot where ROIs with significant positivity or negativity bias are colored differently.

In addition to the quantile regression above, three categorical analyses are run over the same 672 words, with words split into
positive vs negative groups based on the emotion dimensions (positive if positivity >= negativity, else negative):
   A.  Neural within-category coherence: is one valence category represented more similarly to itself (within-group neural
       similarity) than the other? Tested per subject and ROI (contrast = negative - positive).
   B. The same within-category coherence contrast computed on the single LLM similarity matrix (a property of the model,
       so tested with a label-permutation null rather than across subjects).
   C.  Valence-split RSA: is the neural-LLM alignment (Spearman RSA) stronger within positive or within negative word pairs?
       Tested per subject and ROI (contrast = negative - positive).
'''

########################################################################
# Imports
########################################################################
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import wilcoxon, spearmanr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

########################################################################
# Configuration
########################################################################
LLM_NAME = 'BERT'  # Change to your LLM of interest (BERT, ERNIE, Electra, GPT2)

PROJECT_ROOT = Path("/Users/birgitcasselman/Documents/Psychology/Ma2/CaseStudies")  # the only path you need to set
DATA_DIR = PROJECT_ROOT / "data"
BIDS_DIR = DATA_DIR / "ds004301"
DERIVATIVES_DIR = BIDS_DIR / "derivatives"
ANNOTATIONS_DIR = DERIVATIVES_DIR / "annotations"
NEURAL_SIM_DIR = DERIVATIVES_DIR / "similarity_matrices"
DATA_OUTPUT_DIR = DATA_DIR / "rsa_results" / LLM_NAME
VISUAL_OUTPUT_DIR = PROJECT_ROOT / "reports" / "plots" / "rsa_results" / LLM_NAME

LLM_SIM_FILE = ANNOTATIONS_DIR / "embeddings" / "contextual word embeddings" / f"{LLM_NAME}_similarity_matrix_adj.npz"
EMOTION_RATINGS_FILE = DATA_DIR / "emotion_ratings" / "word_ratings.csv"  # Must contain 'word', 'positivity', 'negativity'

N_LABEL_PERMUTATIONS = 10000  # for the B label-permutation test
np.random.seed(42)            # for reproducibility of the label-permutation null

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

########################################################################
# Data loading
########################################################################
llm_matrix_file = np.load(LLM_SIM_FILE, allow_pickle=True)
# Assuming LLM_SIM_FILE actually contains a similarity matrix, not dissimilarity
llm_matrix = distance.squareform(llm_matrix_file['data'])
llm_labels = llm_matrix_file['labels']

emotion_df = pd.read_csv(EMOTION_RATINGS_FILE)
emotion_df = emotion_df.set_index('word').loc[llm_labels].reset_index()

# Get the two independent emotion dimensions (ordered to match llm_labels / the matrices)
pos_scores = emotion_df['positivity'].values
neg_scores = emotion_df['negativity'].values
n_words = len(pos_scores)

########################################################################
# Create Continuous Pairwise Intensity Matrices
########################################################################
pairwise_positivity = np.zeros((n_words, n_words))
pairwise_negativity = np.zeros((n_words, n_words))

for i in range(n_words):
    for j in range(n_words):
        # The intensity of the pair is the average intensity of the two words
        pairwise_positivity[i, j] = (pos_scores[i] + pos_scores[j]) / 2.0
        pairwise_negativity[i, j] = (neg_scores[i] + neg_scores[j]) / 2.0

# Extract upper triangles
triu_indices = np.triu_indices_from(llm_matrix, k=1)
llm_vec = llm_matrix[triu_indices]
pair_pos_vec = pairwise_positivity[triu_indices]
pair_neg_vec = pairwise_negativity[triu_indices]

# Z-score predictors so betas are comparable
llm_z = (llm_vec - np.mean(llm_vec)) / np.std(llm_vec)
pos_z = (pair_pos_vec - np.mean(pair_pos_vec)) / np.std(pair_pos_vec)
neg_z = (pair_neg_vec - np.mean(pair_neg_vec)) / np.std(pair_neg_vec)

# Create interaction terms
interaction_pos = llm_z * pos_z
interaction_neg = llm_z * neg_z

# Design Matrix: X = Constant + Main Effects + Interactions
X = sm.add_constant(np.column_stack([llm_z, pos_z, neg_z, interaction_pos, interaction_neg]))
########################################################################
# RSA via Quantile Regression for each subject and ROI (first level)
########################################################################
neural_sim_files = [os.path.join(NEURAL_SIM_DIR, f, 'similarity_matrices.npz') for f in os.listdir(NEURAL_SIM_DIR) if f.startswith('sub-')]
rsa_results = []

# Create or clear the summary text file before the loop
summary_file_path = os.path.join(DATA_OUTPUT_DIR, 'regression_summaries.txt')
with open(summary_file_path, 'w') as f:
    f.write("Quantile Regression Summaries (q=0.5)\n")
    f.write("="*50 + "\n\n")

for file in tqdm(neural_sim_files, desc="Computing dual emotion-modulated RSA"):
    subject_id = os.path.basename(os.path.dirname(file)).replace('sub-', '')
    data = np.load(file, allow_pickle=True)

    for roi, neural_matrix in data.items():
        neural_vec = neural_matrix[triu_indices]
        neural_z = (neural_vec - np.mean(neural_vec)) / np.std(neural_vec)

        # Fit Quantile Regression for the median (q=0.5)
        model = sm.QuantReg(neural_z, X)
        results = model.fit(q=0.5)

        # Write the summary to the text file instead of printing to the terminal
        with open(summary_file_path, 'a') as f:
            f.write(f"Subject: {subject_id} | ROI: {roi}\n")
            f.write(results.summary().as_text() + "\n\n")

        # Extract interaction beta coefficients [Const, LLM, Pos, Neg, IntPos, IntNeg]
        beta_int_pos = results.params[4]
        beta_int_neg = results.params[5]

        # Store Biases
        rsa_results.append({
            'subject': subject_id,
            'roi': roi,
            'bias_type': f'Positivity Bias ({LLM_NAME} * Positivity)',
            'beta': beta_int_pos
        })
        # Store Negativity Bias
        rsa_results.append({
            'subject': subject_id,
            'roi': roi,
            'bias_type': f'Negativity Bias ({LLM_NAME} * Negativity)',
            'beta': beta_int_neg
        })

rsa_df = pd.DataFrame(rsa_results)
rsa_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'rsa_dual_emotion_bias_results.csv'), index=False)

########################################################################
# Group-Level Statistics (second level, across subjects)
########################################################################
print("\nComputing Group-Level Statistics...")

group_stats = []

for bias in rsa_df['bias_type'].unique():
    for roi in rsa_df['roi'].unique():
        # Get all subject betas for this ROI and this Bias
        subject_betas = rsa_df[(rsa_df['roi'] == roi) & (rsa_df['bias_type'] == bias)]['beta'].values

        # We need at least a few subjects to run a test
        if len(subject_betas) > 3:
            # Wilcoxon signed-rank test (non-parametric test against 0)
            res = wilcoxon(subject_betas - 0)
            p_val = res.pvalue
            stat = res.statistic
        else:
            p_val = np.nan
            stat = np.nan

        group_stats.append({
            'roi': roi,
            'bias_type': bias,
            'mean_beta': np.mean(subject_betas),
            'median_beta': np.median(subject_betas),
            'wilcoxon_stat': stat,
            'p_value_uncorrected': p_val
        })

group_stats_df = pd.DataFrame(group_stats)

# Apply FDR Correction separately for Positivity and Negativity Bias
group_stats_df['p_value_fdr'] = np.nan
for bias in group_stats_df['bias_type'].unique():
    mask = group_stats_df['bias_type'] == bias
    valid_p = group_stats_df[mask]['p_value_uncorrected'].dropna()

    if len(valid_p) > 0:
        _, p_fdr, _, _ = multipletests(valid_p, method='fdr_bh')
        group_stats_df.loc[valid_p.index, 'p_value_fdr'] = p_fdr

# Save stats to CSV
stats_path = os.path.join(DATA_OUTPUT_DIR, 'group_level_statistics.csv')
group_stats_df.to_csv(stats_path, index=False)
print(f"Group level statistics saved to {stats_path}")

# Print significant findings
sig_findings = group_stats_df[group_stats_df['p_value_fdr'] < 0.05]
if not sig_findings.empty:
    print("\n--- ANY Significant ROIs (FDR corrected p < 0.05) ---")
    print(sig_findings[['roi', 'bias_type', 'median_beta', 'p_value_fdr']])
else:
    print("\nNo significant ROIs found at FDR corrected p < 0.05.")

########################################################################
# Visualization
########################################################################
plt.figure(figsize=(18, 8))

# Draw the plot with default colors first
ax = sns.barplot(x='roi', y='beta', hue='bias_type', data=rsa_df, estimator=np.median, errorbar='ci')

# Extract order of X-axis and Hues to know which bar maps to which ROI/Bias
rois = [tick.get_text() for tick in ax.get_xticklabels()]
# Ensure we exactly match the unique hue categories in the order Seaborn plotted them
hue_order = rsa_df['bias_type'].unique()

# Define color schemes
color_map = {
    f'Positivity Bias ({LLM_NAME} * Positivity)': {'sig': "#fe0000", 'non_sig': 'lightgrey'}, # Red / Light Grey
    f'Negativity Bias ({LLM_NAME} * Negativity)': {'sig': "#2100de", 'non_sig': 'darkgrey'}   # Blue / Dark Grey
}

# Iterate through the drawn bar patches (ignoring legend or extraneous patches)
n_rois = len(rois)
bar_patches = [p for p in ax.patches if isinstance(p, Rectangle)]

for i, bar in enumerate(bar_patches):
    # Determine hue group and ROI based on the index
    hue_idx = i // n_rois
    roi_idx = i % n_rois

    # If the index exceeds our expected data patches (e.g., legend patches), stop modifying
    if hue_idx >= len(hue_order) or roi_idx >= len(rois):
        break

    current_hue = hue_order[hue_idx]
    current_roi = rois[roi_idx]

    # Lookup the exact FDR p-value for this ROI and Bias group
    fdr_p = group_stats_df[
        (group_stats_df['roi'] == current_roi) &
        (group_stats_df['bias_type'] == current_hue)
    ]['p_value_fdr'].values

    # Determine significance
    is_sig = False
    if len(fdr_p) > 0 and fdr_p[0] < 0.05:
        is_sig = True

    # Apply coloring
    target_color = color_map[current_hue]['sig'] if is_sig else color_map[current_hue]['non_sig']
    bar.set_facecolor(target_color)

# Clean up baseline and labels
plt.axhline(0, color='black', linestyle='--')
plt.title(f'Does Positivity or Negativity modulate Neural-{LLM_NAME} alignment? (Colored = FDR p < 0.05)')
plt.xlabel('ROI')
plt.ylabel('Interaction Beta Coefficient (Median)')
plt.xticks(rotation=45, ha='right')

# Rebuild legend to clarify colors
legend_handles = [
    mpatches.Patch(color='#d62728', label='Significant Positivity Bias'),
    mpatches.Patch(color='lightgrey', label='Non-sig Positivity Bias'),
    mpatches.Patch(color='#1f77b4', label='Significant Negativity Bias'),
    mpatches.Patch(color='darkgrey', label='Non-sig Negativity Bias')
]
plt.legend(handles=legend_handles, title="Bias Significance")

plt.tight_layout()
plt.savefig(os.path.join(VISUAL_OUTPUT_DIR, 'rsa_dual_emotion_bias.png'))
print("Primary (quantile regression) analysis complete.")

########################################################################
# Categorical valence analyses (A, B, C) over all 672 words
########################################################################
# Split words into positive vs negative groups based on the emotion dimensions.
# A word is positive if positivity >= negativity (ties -> positive), otherwise negative.
is_positive = pos_scores >= neg_scores
pos_idx = np.where(is_positive)[0]
neg_idx = np.where(~is_positive)[0]
print(f"\nValence groups: {len(pos_idx)} positive words, {len(neg_idx)} negative words")


def within_group_mean(matrix, idx):
    '''Mean similarity among all distinct pairs within one group (upper triangle of the group's sub-block).'''
    sub = matrix[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return np.mean(sub[iu])


def split_rsa(neural_matrix, model_matrix, idx):
    '''Spearman RSA between neural and model similarity, restricted to the pairs within one group.'''
    iu = np.triu_indices(len(idx), k=1)
    neural_vals = neural_matrix[np.ix_(idx, idx)][iu]
    model_vals = model_matrix[np.ix_(idx, idx)][iu]
    rho, _ = spearmanr(neural_vals, model_vals)
    return rho, len(iu[0])


def group_wilcoxon(df, contrast_col):
    '''Per-ROI Wilcoxon signed-rank test of a per-subject contrast against 0, with BH-FDR across ROIs.'''
    rows = []
    for roi in df['roi'].unique():
        vals = df[df['roi'] == roi][contrast_col].values
        if len(vals) > 3:
            res = wilcoxon(vals)
            stat, p_val = res.statistic, res.pvalue
        else:
            stat, p_val = np.nan, np.nan
        rows.append({'roi': roi, 'mean_contrast': np.mean(vals), 'median_contrast': np.median(vals),
                     'wilcoxon_stat': stat, 'p_value_uncorrected': p_val})
    out = pd.DataFrame(rows)
    out['p_value_fdr'] = np.nan
    valid_p = out['p_value_uncorrected'].dropna()
    if len(valid_p) > 0:
        _, p_fdr, _, _ = multipletests(valid_p, method='fdr_bh')
        out.loc[valid_p.index, 'p_value_fdr'] = p_fdr
    return out


def plot_contrast(group_df, title, ylabel, fname):
    '''Bar plot of the per-ROI median contrast, colored where FDR-significant.'''
    plt.figure(figsize=(16, 7))
    colors = ['#d62728' if (pd.notna(p) and p < 0.05) else 'lightgrey' for p in group_df['p_value_fdr']]
    plt.bar(range(len(group_df)), group_df['median_contrast'], color=colors)
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(range(len(group_df)), group_df['roi'], rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    legend_handles = [mpatches.Patch(color='#d62728', label='FDR p < 0.05'),
                      mpatches.Patch(color='lightgrey', label='Non-sig')]
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_OUTPUT_DIR, fname), dpi=300)
    plt.close()


# --- A (neural coherence) and C (valence-split RSA), per subject and ROI ---
coherence_results = []
split_rsa_results = []
for file in tqdm(neural_sim_files, desc="Computing valence coherence and split RSA"):
    subject_id = os.path.basename(os.path.dirname(file)).replace('sub-', '')
    data = np.load(file, allow_pickle=True)

    for roi, neural_matrix in data.items():
        # A: within-category neural coherence (contrast = negative - positive)
        within_pos = within_group_mean(neural_matrix, pos_idx)
        within_neg = within_group_mean(neural_matrix, neg_idx)
        coherence_results.append({'subject': subject_id, 'roi': roi,
                                  'within_positive': within_pos, 'within_negative': within_neg,
                                  'contrast_neg_minus_pos': within_neg - within_pos})

        # C: valence-split RSA (contrast = negative - positive)
        rsa_pos, n_pos_pairs = split_rsa(neural_matrix, llm_matrix, pos_idx)
        rsa_neg, n_neg_pairs = split_rsa(neural_matrix, llm_matrix, neg_idx)
        split_rsa_results.append({'subject': subject_id, 'roi': roi,
                                  'rsa_positive': rsa_pos, 'rsa_negative': rsa_neg,
                                  'contrast_neg_minus_pos': rsa_neg - rsa_pos,
                                  'n_positive_pairs': n_pos_pairs, 'n_negative_pairs': n_neg_pairs})

coherence_df = pd.DataFrame(coherence_results)
split_rsa_df = pd.DataFrame(split_rsa_results)
coherence_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'A_neural_coherence_results.csv'), index=False)
split_rsa_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'C_valence_split_rsa_results.csv'), index=False)

# Group-level tests (Wilcoxon + FDR), matching the primary analysis' second-level test
coherence_group = group_wilcoxon(coherence_df, 'contrast_neg_minus_pos')
split_rsa_group = group_wilcoxon(split_rsa_df, 'contrast_neg_minus_pos')
coherence_group.to_csv(os.path.join(DATA_OUTPUT_DIR, 'A_neural_coherence_group_stats.csv'), index=False)
split_rsa_group.to_csv(os.path.join(DATA_OUTPUT_DIR, 'C_valence_split_rsa_group_stats.csv'), index=False)

plot_contrast(coherence_group,
              'A: Neural within-category coherence (negative - positive)',
              'Coherence contrast (neg - pos), median', 'A_neural_coherence.png')
plot_contrast(split_rsa_group,
              f'C: Valence-split Neural-{LLM_NAME} RSA (negative - positive)',
              'RSA contrast (neg - pos), median', 'C_valence_split_rsa.png')

# --- B (model coherence), single matrix, label-permutation test ---
obs_within_pos = within_group_mean(llm_matrix, pos_idx)
obs_within_neg = within_group_mean(llm_matrix, neg_idx)
obs_contrast = obs_within_neg - obs_within_pos

n_pos = len(pos_idx)
null_contrasts = []
for _ in tqdm(range(N_LABEL_PERMUTATIONS), desc=f"B label-permutation ({LLM_NAME})"):
    perm = np.random.permutation(n_words)
    perm_pos, perm_neg = perm[:n_pos], perm[n_pos:]
    null_contrasts.append(within_group_mean(llm_matrix, perm_neg) - within_group_mean(llm_matrix, perm_pos))
null_contrasts = np.array(null_contrasts)
p_two_sided = np.mean(np.abs(null_contrasts) >= np.abs(obs_contrast))

bert_coherence = pd.DataFrame([{
    'model': LLM_NAME,
    'within_positive': obs_within_pos,
    'within_negative': obs_within_neg,
    'contrast_neg_minus_pos': obs_contrast,
    'direction': 'negative more coherent' if obs_contrast > 0 else 'positive more coherent',
    'p_value_label_perm_two_sided': p_two_sided,
    'n_label_permutations': N_LABEL_PERMUTATIONS
}])
bert_coherence.to_csv(os.path.join(DATA_OUTPUT_DIR, f'B_model_coherence_{LLM_NAME}.csv'), index=False)
print(f"\nB ({LLM_NAME}) within-category coherence contrast (neg - pos): {obs_contrast:.4f}, "
      f"two-sided label-permutation p = {p_two_sided:.4f}")

# Plot the label-permutation null with the observed contrast
plt.figure(figsize=(8, 6))
sns.histplot(null_contrasts, bins=40, kde=True)
plt.axvline(obs_contrast, color='red', linestyle='--', label='Observed (neg - pos)')
plt.title(f'B: {LLM_NAME} within-category coherence (label-permutation null)')
plt.xlabel('Coherence contrast (neg - pos)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_OUTPUT_DIR, f'B_model_coherence_{LLM_NAME}.png'), dpi=300)
plt.close()

print("All analyses complete. Results saved to output directory.")

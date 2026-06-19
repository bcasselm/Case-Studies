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

The goal of this analysis is to determine if certain brain regions show a stronger alignment between neural and LLM similarity 
for words that are more positive or more negative, which would suggest an emotion-specific modulation of semantic representations in LLMs.
'''

########################################################################
# Imports
########################################################################
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

########################################################################
# Configuration 
########################################################################
LLM_NAME = 'GPT2' # Change to your LLM of interest (BERT, ERNIE, Electra, GTP2)

DATA_DIR = '/home/f_moldovan/projects/case_studies/data'
BIDS_DIR = os.path.join(DATA_DIR, 'bids')
DERIVATIVES_DIR = os.path.join(BIDS_DIR, 'derivatives')
ANNOTATIONS_DIR = os.path.join(DERIVATIVES_DIR, 'annotations')
NEURAL_SIM_DIR = os.path.join(DERIVATIVES_DIR, 'similarity_matrices')
DATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'rsa_results', LLM_NAME)
VISUAL_OUTPUT_DIR = f'/home/f_moldovan/projects/case_studies/reports/plots/rsa_results/{LLM_NAME}'

LLM_SIM_FILE = os.path.join(ANNOTATIONS_DIR, f'embeddings/contextual word embeddings/{LLM_NAME}_similarity_matrix_adj.npz')
EMOTION_RATINGS_FILE = os.path.join(DATA_DIR, 'emotion_ratings/word_ratings.csv') # Must contain 'word', 'positivity', 'negativity'

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

np.random.seed(42)  # For reproducibility of bootstrapped confidence intervals

########################################################################
# Helper functions
########################################################################
def load_and_prepare_data(llm_sim_file, emotion_ratings_file):
    """Loads similarity matrix and emotion ratings, returning aligned structures."""
    llm_matrix_file = np.load(llm_sim_file, allow_pickle=True)

    # Assuming LLM_SIM_FILE actually contains a similarity matrix, not dissimilarity
    llm_matrix = distance.squareform(llm_matrix_file['data'])
    llm_labels = llm_matrix_file['labels']

    emotion_df = pd.read_csv(emotion_ratings_file)
    print(llm_labels == emotion_df['word'].values)  # Check if the order of words matches (alignment or not) -> it does

    # Get the two independent emotion dimensions
    pos_scores = emotion_df['positivity'].values
    neg_scores = emotion_df['negativity'].values
    
    return llm_matrix, pos_scores, neg_scores

def create_design_matrix(llm_matrix, pos_scores, neg_scores):
    """Generates continuous pairwise intensity matrices and the design matrix X."""
    n_words = len(pos_scores)
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
    
    return X, triu_indices

def run_quantile_regression_rsa(neural_sim_dir, triu_indices, X, llm_name, data_output_dir):
    """Loops through subjects/ROIs and fits Quantile Regression to derive beta coefficients."""
    neural_sim_files = [os.path.join(neural_sim_dir, f, 'similarity_matrices.npz') for f in os.listdir(neural_sim_dir) if f.startswith('sub-')]
    rsa_results = []

    # Create or clear the summary text file before the loop
    summary_file_path = os.path.join(data_output_dir, 'regression_summaries.txt')
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

            # Store Positivity Bias
            rsa_results.append({
                'subject': subject_id,
                'roi': roi,
                'bias_type': f'Positivity Bias ({llm_name} * Positivity)',
                'beta': beta_int_pos
            })
            # Store Negativity Bias
            rsa_results.append({
                'subject': subject_id,
                'roi': roi,
                'bias_type': f'Negativity Bias ({llm_name} * Negativity)',
                'beta': beta_int_neg
            })

    rsa_df = pd.DataFrame(rsa_results)
    rsa_df.to_csv(os.path.join(data_output_dir, 'rsa_dual_emotion_bias_results.csv'), index=False)
    
    return rsa_df

def compute_group_level_statistics(rsa_df, data_output_dir):
    """Computes second level group statistics via Wilcoxon tests + FDR corrections."""
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
    stats_path = os.path.join(data_output_dir, 'group_level_statistics.csv')
    group_stats_df.to_csv(stats_path, index=False)
    print(f"Group level statistics saved to {stats_path}")

    # Print significant findings
    sig_findings = group_stats_df[group_stats_df['p_value_fdr'] < 0.05]
    if not sig_findings.empty:
        print("\n--- ANY Significant ROIs (FDR corrected p < 0.05) ---")
        print(sig_findings[['roi', 'bias_type', 'median_beta', 'p_value_fdr']])
    else:
        print("\nNo significant ROIs found at FDR corrected p < 0.05.")
        
    return group_stats_df

def visualize_rsa_results(rsa_df, group_stats_df, llm_name, visual_output_dir):
    """Draws and saves the significance-colored statistical bar plot."""
    plt.figure(figsize=(18, 8))

    # Draw the plot with default colors first
    ax = sns.barplot(x='roi', y='beta', hue='bias_type', data=rsa_df, estimator=np.median, errorbar='ci', n_boot = 10000) # 10000 bootstrap samples for 95% CI

    # Extract order of X-axis and Hues to know which bar maps to which ROI/Bias
    rois = [tick.get_text() for tick in ax.get_xticklabels()]
    # Ensure we exactly match the unique hue categories in the order Seaborn plotted them
    hue_order = rsa_df['bias_type'].unique()

    # Define color schemes
    color_map = {
        f'Positivity Bias ({llm_name} * Positivity)': {'sig': "#fe0000", 'non_sig': 'lightgrey'}, # Red / Light Grey
        f'Negativity Bias ({llm_name} * Negativity)': {'sig': "#2100de", 'non_sig': 'darkgrey'}   # Blue / Dark Grey
    }

    # Iterate through the drawn bar patches (ignoring legend or extraneous patches)
    n_rois = len(rois)
    from matplotlib.patches import Rectangle
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
    plt.title(f'Does Positivity or Negativity modulate Neural-{llm_name} alignment? (Colored = FDR p < 0.05)')
    plt.xlabel('ROI')
    plt.ylabel('Interaction Beta Coefficient (Median)')
    plt.xticks(rotation=45, ha='right')

    # Rebuild legend to clarify colors
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#d62728', label='Significant Positivity Bias'),
        mpatches.Patch(color='lightgrey', label='Non-sig Positivity Bias'),
        mpatches.Patch(color='#1f77b4', label='Significant Negativity Bias'),
        mpatches.Patch(color='darkgrey', label='Non-sig Negativity Bias')
    ]
    plt.legend(handles=legend_handles, title="Bias Significance")

    plt.tight_layout()
    plt.savefig(os.path.join(visual_output_dir, 'rsa_dual_emotion_bias.png'))
    print("All analyses complete. Results saved to output directory.")

########################################################################
# Main execution
########################################################################
if __name__ == "__main__":
    # 1. Load data
    llm_matrix, pos_scores, neg_scores = load_and_prepare_data(LLM_SIM_FILE, EMOTION_RATINGS_FILE)
    
    # 2. Setup regression design matrices
    X, triu_indices = create_design_matrix(llm_matrix, pos_scores, neg_scores)
    
    # 3. Perform RSA via Quantile Regression
    rsa_df = run_quantile_regression_rsa(NEURAL_SIM_DIR, triu_indices, X, LLM_NAME, DATA_OUTPUT_DIR)
    
    # 4. Compute second-level group statistics
    group_stats_df = compute_group_level_statistics(rsa_df, DATA_OUTPUT_DIR)
    
    # 5. Visualize Modulated Alignments
    visualize_rsa_results(rsa_df, group_stats_df, LLM_NAME, VISUAL_OUTPUT_DIR)
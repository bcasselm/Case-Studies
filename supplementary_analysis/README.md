# Supplementary Analysis — Representational Similarity Analysis of fMRI Data on Emotional Concept Representation and LLM-Derived Contextual Embeddings

This is the supplementary analysis pipeline accompanying the main analysis. It builds on the main analysis and reuses most of its code, addressing the same research question: how similar is the brain's representation of concepts is to representations derived from large language models, and whether this similarity varies with the emotional valence of the words (and across brain regions). 

It differs from the main analysis in two ways: it uses an alternative approach to estimating the first-level beta maps, and it adds an alternative way to conceptualize the emotional-valence question through several categorical valence analyses. The aim is to examine whether the conclusions of the main analysis also hold under these alternative analytical choices.

All data used in the project is publicly available and can be accessed via the OpenNeuro dataset [ds004301](https://doi.org/10.18112/openneuro.ds004301.v1.0.2) ([Wang et al., 2023](https://doi.org/10.1038/s41597-022-01840-2)).

## Key differences from the main pipeline

- **First-level GLM (`02`):** beta maps are estimated with a Least-Squares-Separate (LSS) approach, fitting a separate GLM per trial (the target trial modelled on its own, all other trials collapsed into a single regressor) and then averaging a word's single-trial betas into one beta per word, instead of the main pipeline's single combined GLM. This is accompanied by the following modelling choices: no spatial smoothing (to preserve fine-grained multivariate patterns), an AR(1) noise model, explicit percent-signal-change scaling, high-pass filtering applied via cosine regressors taken from fMRIPrep's confounds file, and a confound set of 24 motion parameters, 6 aCompCor components, and spike/non-steady-state regressors.

- **Emotion-modulated RSA (`07`):** the main pipeline's interaction-term quantile regression is retained as the primary analysis, and three additional categorical valence analyses are added over all 672 words, with words split into positive, neutral, and negative groups using a difference-score threshold δ = 1.0 (positive if positivity − negativity > δ, negative if negativity − positivity > δ, neutral otherwise): **(A)** neural within-category coherence — the mean neural similarity within each valence group, with all three pairwise contrasts (neg−pos, neg−neu, pos−neu) tested across subjects per ROI (Wilcoxon signed-rank with BH-FDR correction); **(B)** model within-category coherence — the same within-group coherence computed on the single LLM similarity matrix, with all three pairwise contrasts tested via a label-permutation null (10,000 permutations); and **(C)** valence-split RSA — the brain-LLM Spearman correlation computed separately within each valence group, with all three pairwise contrasts tested across subjects per ROI (Wilcoxon signed-rank with BH-FDR correction).

## Contents
This repository contains all the code used for the abovementioned analyses, and is organized as follows:
- `01_brain_parcellation.py`: Code for parcellating the brain into regions of interest (ROIs) using a coordinate-based meta-analytic approach (Neurosynth) with cluster-mass based correction.

- `02_first_level_GLMs.py`: Code for fitting first-level general linear models (GLMs) to the fMRI data to obtain beta estimates for each word condition for each subject in each ROI. **Change:** uses Least-Squares-Separate (LSS) single-trial estimation, then averages each word's repetitions into one beta per word (see Key differences).

- `03_neural_similarities.py`: Code for computing neural similarity matrices (RSMs) for each subject and ROI based on the beta estimates obtained from the first-level GLMs, using pairwise Pearson correlations.

- `04_LLM_similarity.py`: Code for computing LLM-derived similarity matrices (RSMs) for the same set of words using contextual embeddings from a large language model (LLM) of choice (BERT, ERNIE, Electra, or GPT2). Cosine similarity is used to compute the similarity between word embeddings.

- `05_rsa_general.py`: Code for performing representational similarity analysis (RSA) by correlating the neural RSMs with the LLM-derived RSMs for each subject and ROI, and assessing statistical significance across subjects for each ROI (and across all the ROIs) using permutation testing (and bootstrapped confidence intervals).

- `06_emotion_dimensions.py`: Code for quantifying the emotional valence of each word using human ratings on several emotion features. We aggregate (median) these ratings into two main dimensions of emotional valence: positivity and negativity.

- `07_rsa_emotion.py`: Code for performing RSA with interaction terms to test whether the brain-LLM similarity is modulated by emotional valence (positivity and negativity). We fit a quantile regression model for each subject and ROI, where the dependent variable is the neural similarity (vectorized upper triangle of the neural RSM), and the independent variables are the LLM similarity (vectorized upper triangle of the LLM RSM), the positivity and negativity ratings (for each word pair, essentially vectorized upper triangles of positive and negative emotion rating matrices), and their interactions. We then extract the beta coefficients for the interaction terms to assess whether there is a significant modulation of brain-LLM similarity by emotional valence across subjects for each ROI, visualized in a bar plot of the median interaction beta coefficients per ROI, colored by statistical significance after FDR correction. **Change:** in addition to this primary analysis, three categorical valence analyses over all 672 words with a three-way valence split (positive / neutral / negative, δ = 1.0) are run (see Key differences).

Additionally, the `reports` folder contains all visualizations (examples and results) generated from the above analyses.

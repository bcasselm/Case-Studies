# Representational Similarity Analysis of fMRI Data on Emotional Concept Representation and LLM-Derived Contextual Embeddings

This project focuses on the representational similarity analysis of fMRI data on emotional concept (word) representation and LLM-derived contextual embeddings of these words. We aim to understand how similar the brain's representation of concepts is to the representations derived from large language models, and how this similarity varies according to the emotional valence of these words (and across different brain regions). Specifically, we investigate whether the brain's representation of emotional concepts is more similar to LLM-derived representations for negatively valenced words compared to positively valenced words, and whether this relationship is stronger in certain brain regions (e.g., amygdala, insula) known to be involved in emotional processing. 

All data used in the project is publicly available and can be accessed via the OpenNeuro dataset [ds004301](https://doi.org/10.18112/openneuro.ds004301.v1.0.2) ([Wang et al., 2023](https://doi.org/10.1038/s41597-022-01840-2)). 

## Contents
This repository contains all the code (`source_code`) used for the abovementioned analyses, and is organized as follows:
- `01_brain_parcellation.py`: Code for parcellating the brain into regions of interest (ROIs) using a coordinate-based meta-analytic approach (Neurosynth) with cluster-mass based correction. 

- `02_first_level_GLMS.py`: Code for fitting first-level general linear models (GLMs) to the fMRI data to obtain beta estimates for each word condition for each subject in each ROI.

- `03_neural_similarities.py`: Code for computing neural similarity matrices (RSMs) for each subject and ROI based on the beta estimates obtained from the first-level GLMs, using pairwise Pearson correlations.

- `04_LLM_similarity.py`: Code for computing LLM-derived similarity matrices (RSMs) for the same set of words using contextual embeddings from a large language model (LLM) of choice (BERT, ERNIE, Electra, or GPT2). Cosine similarity is used to compute the similarity between word embeddings.

- `05_rsa_general.py`: Code for performing representational similarity analysis (RSA) by correlating the neural RSMs with the LLM-derived RSMs for each subject and ROI, and assessing statistical significance across subjects for each ROI (and across all the ROIs) using permutation testing (and bootstrapped confidence intervals). However, this RSA approach does not account for the emotional valence of the words, and thus does not allow us to test whether the brain-LLM similarity is modulated by emotional valence. This is done by the next two scripts. 

- `06_emotion_dimensions.py`: Code for quantifying the emotional valence of each word using human ratings on several emotion features. We aggegate (median) these ratings into two main dimensions of emotional valence: positivity and negativity.

- `07_rsa_emotion.py`: Code for performing RSA with interaction terms to test whether the brain-LLM similarity is modulated by emotional valence (positivity and negativity). We fit a quantile regression model for each subject and ROI, where the dependent variable is the neural similarity (vectorized upper triangle of the neural RSM), and the independent variables are the LLM similarity (vectorized upper triangle of the LLM RSM), the positivity and negativity ratings (for each word pair, essentially vectorized upper triangles of positive and negative emotion rating matrices), and their interactions. We then extract the beta coefficients for the interaction terms to assess whether there is a significant modulation of brain-LLM similarity by emotional valence across subjects for each ROI. Finally, we visualize these results in a bar plot showing the median interaction beta coefficients for each ROI (across subjects), colored by statistical significance after FDR correction.

Additionally, the `reports` folder contains all visualizations (examples and results) generated from the above analyses, and the `supplementary_analysis` folder contains any additional analyses or figures that support the main findings.
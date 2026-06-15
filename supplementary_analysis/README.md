# Supplementary Analysis — Representational Similarity Analysis of fMRI Data on Emotional Concept Representation and LLM-Derived Contextual Embeddings

This is the supplementary analysis pipeline accompanying the main analysis. It builds on the main analysis and reuses most of its code, addressing the same research question: how similar is the brain's representation of concepts is to representations derived from large language models, and whether this similarity varies with the emotional valence of the words (and across brain regions). 

It differs from the main analysis in two ways: it uses an alternative approach to estimating the first-level beta maps, and it adds an alternative way to operationalize the emotional-valence question through several categorical valence analyses. The aim is to examine whether the conclusions of the main analysis also hold under these alternative analytical choices.

All data used in the project is publicly available and can be accessed via the OpenNeuro dataset [ds004301](https://doi.org/10.18112/openneuro.ds004301.v1.0.2) ([Wang et al., 2023](https://doi.org/10.1038/s41597-022-01840-2)).

## Key differences from the main pipeline

- **First-level GLM (`02`):** beta maps are estimated with a Least-Squares-Separate (LSS) approach, fitting a separate GLM per trial (the target trial modelled on its own, all other trials collapsed into a single regressor) and then averaging a word's single-trial betas into one beta per word, instead of the main pipeline's single combined GLM. This is accompanied by the modelling choices that suit single-trial pattern estimation: no spatial smoothing (to preserve fine-grained multivariate patterns), an AR(1) noise model, explicit percent-signal-change scaling, high-pass filtering applied once via the cosine drift model, and a confound set of 24 motion parameters, 6 aCompCor components, and spike/non-steady-state regressors.

- **Neural similarity matrices (`03`):** before correlating, the cross-condition mean pattern is removed from each ROI ("cocktail-blank" removal), so the ROI-wide average response shared across conditions does not inflate the similarities. The metric is otherwise the same pairwise Pearson correlation as in the main pipeline.

- **Emotion-modulated RSA (`07`):** the main pipeline's interaction-term quantile regression is retained as the primary analysis, and three additional categorical valence analyses are added over all 672 words, with words split into positive and negative groups based on the emotion dimensions (positive if positivity ≥ negativity, else negative): **(A)** neural within-category coherence — whether one valence category is represented more similarly to itself than the other in the neural patterns; **(B)** model within-category coherence — the same contrast computed on the single LLM similarity matrix, tested with a label-permutation null; and **(C)** valence-split RSA — whether the brain-LLM alignment is stronger within positive or within negative word pairs. Analyses A and C are tested across subjects per ROI (Wilcoxon signed-rank with FDR correction), matching the primary analysis.

## Contents
This repository contains all the code used for the abovementioned analyses, and is organized as follows:
- `01_brain_parcellation.py`: Code for parcellating the brain into regions of interest (ROIs) using a coordinate-based meta-analytic approach (Neurosynth) with cluster-mass based correction.

- `02_first_level_GLMs.py`: Code for fitting first-level general linear models (GLMs) to the fMRI data to obtain beta estimates for each word condition for each subject in each ROI. **Change:** uses Least-Squares-Separate (LSS) single-trial estimation, then averages each word's repetitions into one beta per word (see Key differences).

- `03_neural_similarities.py`: Code for computing neural similarity matrices (RSMs) for each subject and ROI based on the beta estimates obtained from the first-level GLMs, using pairwise Pearson correlations. **Change:** applies cocktail-blank (cross-condition mean pattern) removal within each ROI before correlating.

- `04_LLM_similarity.py`: Code for computing LLM-derived similarity matrices (RSMs) for the same set of words using contextual embeddings from a large language model (LLM) of choice (BERT, ERNIE, Electra, or GPT2). Cosine similarity is used to compute the similarity between word embeddings.

- `05_rsa_general.py`: Code for performing representational similarity analysis (RSA) by correlating the neural RSMs with the LLM-derived RSMs for each subject and ROI, and assessing statistical significance across subjects for each ROI (and across all the ROIs) using permutation testing (and bootstrapped confidence intervals).

- `06_emotion_dimensions.py`: Code for quantifying the emotional valence of each word using human ratings on several emotion features. We aggregate (median) these ratings into two main dimensions of emotional valence: positivity and negativity.

- `07_rsa_emotion.py`: Code for performing RSA with interaction terms to test whether the brain-LLM similarity is modulated by emotional valence (positivity and negativity). We fit a quantile regression model for each subject and ROI, where the dependent variable is the neural similarity (vectorized upper triangle of the neural RSM), and the independent variables are the LLM similarity (vectorized upper triangle of the LLM RSM), the positivity and negativity ratings (for each word pair, essentially vectorized upper triangles of positive and negative emotion rating matrices), and their interactions. We then extract the beta coefficients for the interaction terms to assess whether there is a significant modulation of brain-LLM similarity by emotional valence across subjects for each ROI, visualized in a bar plot of the median interaction beta coefficients per ROI, colored by statistical significance after FDR correction. **Change:** in addition to this primary analysis, three categorical valence analyses are run over all 672 words (see Key differences).

Additionally, the `reports` folder contains all visualizations (examples and results) generated from the above analyses.

## Configuration and run order
Each script anchors all of its paths to a single `PROJECT_ROOT` constant at the top of the file. Set this once per script to the folder containing the `data/` directory. The scripts expect the dataset at `data/ds004301` (with fMRIPrep derivatives under `derivatives/preprocessed_data`), and the 11 subjects (`sub-01`…`sub-11`) are listed explicitly in each script.

The scripts are run in numerical order (`01` → `07`), with the following dependencies: `01` produces the parcellation mask read by `02`, `03`, and `07`; `02` produces the per-word betas read by `03`; `03` produces the neural RSMs read by `05` and `07`; `04` produces the LLM RSM read by `05` and `07`; and `06` produces the emotion dimensions read by `07`.

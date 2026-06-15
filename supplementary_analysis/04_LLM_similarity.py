'''
Computes the similarity matrix between the LLM contextual word embeddings for the 672 words in our stimulus set.
The similarity metric used is cosine similarity, which is commonly used for word embeddings.
The resulting similarity matrix is stored as an Adjacency object from nltools, which can be easily manipulated and visualized later on.
'''

#####################################################################
# Imports
#####################################################################
import os
from pathlib import Path
import scipy.io
from scipy.spatial import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nltools.data import Adjacency

#####################################################################
# Configuration
#####################################################################
LLM_NAME = 'BERT'  # which LLM embeddings to use: 'BERT', 'ERNIE', 'Electra', or 'GPT2'

PROJECT_ROOT = Path("/Users/birgitcasselman/Documents/Psychology/Ma2/CaseStudies")  # the only path you need to set
BIDS_DIR = PROJECT_ROOT / "data" / "ds004301"
ANNOTATIONS_DIR = BIDS_DIR / "derivatives" / "annotations"
EMBEDDINGS_PATH = ANNOTATIONS_DIR / "embeddings" / "contextual word embeddings" / f"{LLM_NAME}.mat"  # 672 embeddings for the chosen LLM
TRANSLATIONS_FILE_PATH = ANNOTATIONS_DIR / "672words_translations.csv"  # maps Chinese words to their English translations
OUT_DIR = ANNOTATIONS_DIR / "embeddings" / "contextual word embeddings"  # where the similarity matrix is saved
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots" / "embedding_sim_matrix"  # where the plot is saved

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

#####################################################################
# Data loading
#####################################################################
embeddings = scipy.io.loadmat(str(EMBEDDINGS_PATH))
print("LLM embeddings keys:", embeddings.keys())
print(embeddings['data'].shape)  # Should be (672, 769): 768 embedding dims + first column with word labels

labels = np.squeeze(embeddings['data'][:, 0])                       # word labels (first column)
labels = np.vectorize(lambda x: str(np.squeeze(x)))(labels)        # convert from 1x1 cell arrays to strings
print("Sample word labels (Chinese):", labels[:10])
data = np.asarray(embeddings['data'][:, 1:])                       # embeddings (ignore first column)

# Translate the word labels from Chinese to English
translations = pd.read_csv(TRANSLATIONS_FILE_PATH, header=None, encoding="utf-8-sig")
english_labels = []
for chinese in labels:
    translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
    english_labels.append(translation)
print("Sample English labels:", english_labels[:10])

# Entries are 1x1 cell arrays, so we extract the actual values
embeddings = np.vectorize(lambda x: float(np.squeeze(x)))(data)
print(embeddings)

#####################################################################
# Compute cosine similarity matrix between the contextual word
# embeddings for the 672 words
#####################################################################
similarity_matrix = 1 - distance.pdist(embeddings, metric='cosine')
similarity_matrix = distance.squareform(similarity_matrix)  # convert to square form
print(f"{LLM_NAME} similarity matrix shape:", similarity_matrix.shape)  # Should be (672, 672)

#######################################################################
# Visualize and save the similarity matrix as png file
#######################################################################
plotting.plot_matrix(similarity_matrix,
                     labels=english_labels,
                     colorbar=True,
                     title=f'{LLM_NAME} Contextual Word Embeddings Similarity Matrix (cosine similarity)',
                     cmap='viridis')
plt.savefig(os.path.join(PLOTS_DIR, f'{LLM_NAME}_similarity_matrix.png'), dpi=300, bbox_inches='tight')

########################################################################
# Store the similarity matrix as an Adjacency object from nltools
# (square (672, 672) form, with the English labels)
########################################################################
adjacency = Adjacency(similarity_matrix, labels=english_labels)
np.savez_compressed(
    os.path.join(OUT_DIR, f'{LLM_NAME}_similarity_matrix_adj.npz'),
    data=adjacency.data,
    labels=np.array(adjacency.labels, dtype=object)
)

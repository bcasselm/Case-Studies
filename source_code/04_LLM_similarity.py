'''
Computes the similarity matrix between the LLM contextual word embeddings for the 672 words in our stimulus set.
The similarity metric used is cosine similarity, which is commonly used for word embeddings.
The resulting similarity matrix is stored as an Adjacency object from nltools, which can be easily manipulated and visualized later on.
'''

#####################################################################
# Imports
#####################################################################
import os
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
LLM_NAME = 'BERT' # Change this to the name of the LLM you are using (Electra, GPT2, ERNIE, BERT) to keep track of which similarity matrix corresponds to which LLM embeddings

BIDS_DIR = '/home/f_moldovan/projects/case_studies/data/bids'
EMBEDDINGS_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', 'embeddings', 'contextual word embeddings', f'{LLM_NAME}.mat') # Path to the .mat file containing the embeddings for the 672 words, change this to the other embeddings if you want to compute similarity matrices for the other types of embeddings (Electra, GPT2, ERNIE)
TRANSLATIONS_FILE_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', '672words_translations.csv') # This file contains the mapping from Chinese words to their English translations
OUT_DIR = os.path.join(BIDS_DIR, 'derivatives', 'annotations', 'embeddings', 'contextual word embeddings') # Directory where we will save the computed similarity matrix as an Adjacency object
PLOTS_DIR = '/home/f_moldovan/projects/case_studies/reports/plots/embedding_sim_matrix' # Directory to save the plot of the similarity matrix

#####################################################################
# Helper functions
#####################################################################
def load_and_preprocess_embeddings(embeddings_path, translations_path):
    """Loads .mat embeddings, extracts labels/data, and translates to English."""
    embeddings = scipy.io.loadmat(embeddings_path)
    print("LLM embeddings keys:", embeddings.keys())
    print(embeddings['data'].shape)  # Should be (672, 769 or 257), 768 (BERT, ERNIE, GPT2) or 256 (Electra) is the dimensionality of LLM embeddings for each word and the first column is the word labels (672 words)

    labels = np.squeeze(embeddings['data'][:, 0])  # Extract the word labels (first column)
    labels = np.vectorize(lambda x: str(np.squeeze(x)))(labels)  # Convert from 1x1 cell arrays to strings
    print("Sample word labels (Chinese):", labels[:10])  # Print the first 10 word labels to check if they are in Chinese as expected
    data = np.asarray(embeddings['data'][:, 1:]) # Extract the embeddings, ignoring the first column with word labels

    # Translate the word labels from Chinese to English using the 672words_translations.csv file
    translations = pd.read_csv(translations_path, header=None)
    english_labels = []
    for chinese in labels:
        translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
        english_labels.append(translation) # builiding list of 672 English words corresponding to the 672 conditions
    print("Sample English labels:", english_labels[:10]) # same order as in labels list

    # Entries are 1x1 cell arrays, so we need to extract the actual values
    embeddings_data = np.vectorize(lambda x: float(np.squeeze(x)))(data)
    print(embeddings_data)
    
    return embeddings_data, english_labels

def compute_similarity_matrix(embeddings_data, llm_name):
    """Computes cosine similarity matrix between the contextual word embeddings."""
    # Compute cosine similarity matrix between the BERT contextual word 
    # embeddings for the 672 words
    similarity_matrix = 1 - distance.pdist(embeddings_data, metric='cosine')
    similarity_matrix = distance.squareform(similarity_matrix)  # Convert to square form
    print(f"{llm_name} similarity matrix shape:", similarity_matrix.shape)  # Should be (672, 672)
    
    return similarity_matrix

def plot_and_save_matrix(similarity_matrix, english_labels, llm_name, plots_dir):
    """Visualizes and saves the similarity matrix as a PNG file."""
    os.makedirs(plots_dir, exist_ok=True)
    im = plotting.plot_matrix(similarity_matrix, 
                         labels=english_labels, 
                         colorbar=True, 
                         title='',
                         cmap='viridis')  # Diverging colormap to show positive and negative similarities
    
    ax = im.axes

    ax.set_xticklabels(labels=english_labels, rotation=90)
    ax.set_yticklabels(labels=english_labels, rotation=0)
    plt.savefig(os.path.join(plots_dir, f'{llm_name}_similarity_matrix.png'), dpi=300, bbox_inches='tight')

def save_adjacency_matrix(similarity_matrix, english_labels, llm_name, out_dir):
    """Stores the similarity matrix as an Adjacency object from nltools and saves to NPZ."""
    # Store the similarity matrix as an Adjacency object from nltools
    # But ensure its shape is (672, 672) and not flattened, and also store 
    # the labels in the Adjacency object
    os.makedirs(out_dir, exist_ok=True)
    adjacency = Adjacency(similarity_matrix, labels=english_labels)
    np.savez_compressed(
        os.path.join(out_dir, f'{llm_name}_similarity_matrix_adj.npz'),
        data=adjacency.data,
        labels=np.array(adjacency.labels, dtype=object)
    )


#####################################################################
# Main execution
#####################################################################
if __name__ == "__main__":
    # 1. Load data and translate labels
    embeddings_data, english_labels = load_and_preprocess_embeddings(EMBEDDINGS_PATH, TRANSLATIONS_FILE_PATH)
    
    # 2. Compute similarity
    similarity_matrix = compute_similarity_matrix(embeddings_data, LLM_NAME)
    
    # 3. Plot and save visualization
    plot_and_save_matrix(similarity_matrix, english_labels, LLM_NAME, PLOTS_DIR)
    
    # 4. Save to NPZ for later stages
    save_adjacency_matrix(similarity_matrix, english_labels, LLM_NAME, OUT_DIR)
'''
Computes the similarity matrix between the BERT contextual word embeddings for the 672 words in our stimulus set.
The similarity metric used is cosine similarity, which is commonly used for word embeddings.
The resulting similarity matrix is stored as an Adjacency object from nltools, which can be easily manipulated and visualized later on.
'''

#####################################################################
# Imports
#####################################################################
import scipy.io
from scipy.spatial import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nltools.data import Adjacency

#####################################################################
# Data loading
#####################################################################
BERT_embeddings = scipy.io.loadmat('/home/f_moldovan/projects/case_studies/data/bids/derivatives/annotations/embeddings/contextual word embeddings/BERT.mat')
print("BERT embeddings keys:", BERT_embeddings.keys())
print(BERT_embeddings['data'].shape)  # Should be (672, 769), 768 is the dimensionality of BERT embeddings for each word and the first column is the word labels (672 words)

labels = np.squeeze(BERT_embeddings['data'][:, 0])  # Extract the word labels (first column)
labels = np.vectorize(lambda x: str(np.squeeze(x)))(labels)  # Convert from 1x1 cell arrays to strings
print("Sample word labels (Chinese):", labels[:10])  # Print the first 10 word labels to check if they are in Chinese as expected
data = np.asarray(BERT_embeddings['data'][:, 1:]) # Extract the embeddings, ignoring the first column with word labels

# Translate the word labels from Chinese to English using the 672words_translations.csv file
translations = pd.read_csv("/home/f_moldovan/projects/case_studies/data/bids/derivatives/annotations/672words_translations.csv", header=None)
english_labels = []
for chinese in labels:
    translation = translations[translations.iloc[:, 0] == chinese].iloc[0, 1]
    english_labels.append(translation) # builiding list of 672 English words corresponding to the 672 conditions
print("Sample English labels:", english_labels[:10]) # same order as in labels list

# Entries are 1x1 cell arrays, so we need to extract the actual values
embeddings = np.vectorize(lambda x: float(np.squeeze(x)))(data)
print(embeddings)

#####################################################################
# Compute cosine similarity matrix between the BERT contextual word 
# embeddings for the 672 words
#####################################################################
similarity_matrix = 1 - distance.pdist(embeddings, metric='cosine')
similarity_matrix = distance.squareform(similarity_matrix)  # Convert to square form
print("BERT similarity matrix shape:", similarity_matrix.shape)  # Should be (672, 672)

#######################################################################
# Visualize and save the similarity matrix as png file
#######################################################################
plotting.plot_matrix(similarity_matrix, 
                     labels=english_labels, 
                     colorbar=True, 
                     title='BERT Contextual Word Embeddings Similarity Matrix (cosine similarity)',
                     cmap='viridis')  # Diverging colormap to show positive and negative similarities
plt.savefig('/home/f_moldovan/projects/case_studies/reports/plots/embedding_sim_matrix/BERT_similarity_matrix.png', dpi=300, bbox_inches='tight')

########################################################################
# Store the similarity matrix as an Adjacency object from nltools
# But ensure its shape is (672, 672) and not flattened, and also store 
# the labels in the Adjacency object
########################################################################
adjacency = Adjacency(similarity_matrix, labels=english_labels)
np.savez_compressed(
    '/home/f_moldovan/projects/case_studies/data/bids/derivatives/annotations/embeddings/contextual word embeddings/BERT_similarity_matrix_adj.npz',
    data=adjacency.data,
    labels=np.array(adjacency.labels, dtype=object)
)
'''
Computes the negativity and positivity rating for each of the 672 words.

Specifically, it takes the median ratings of Benefit and Happy for positivity, 
and the median ratings of Sad, Angry, Disgusted, and Fearful for negativity. 

Then, it stores these ratings in a CSV file, which can be used later for 
emotion-specific RSA analyses to test for positivity or negativity bias in the 
RSA between human and LLM word representations.
'''

######################################################################
# Imports
######################################################################
import os
import numpy as np
import pandas as pd

######################################################################
# Configuration
######################################################################
DATA_DIR = '/home/f_moldovan/projects/case_studies/data'
BIDS_DIR = os.path.join(DATA_DIR, 'bids')
RATINGS_FILE_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', 'semantic feature', 'feature.csv')
TRANSLATIONS_FILE_PATH = os.path.join(BIDS_DIR, 'derivatives', 'annotations', '672words_translations.csv')
OUT_DIR = os.path.join(DATA_DIR, 'emotion_ratings')
os.makedirs(OUT_DIR, exist_ok=True)

######################################################################
# Data loading
######################################################################
ratings_df = pd.read_csv(RATINGS_FILE_PATH)

print(ratings_df.head())  # Check the structure of the ratings data
print(ratings_df.columns)  # Check column names to identify relevant features 
# Columns are all in Chinese but the order of them corresponds to the order of the English translations in the authors paper, 
# so we can identify the relevant columns based on their order (with first column being the word label):
# Benefit (column 45), Happy (column 46), Sad (column 47), Angry (column 48), Disgusted (column 49), Fearful (column 50)

######################################################################
# Compute positivity and negativity ratings for each word
######################################################################
word_ratings = pd.DataFrame()
for index, row in ratings_df.iterrows():
    word = row[0]  # Assuming the first column contains the word label
    positivity = np.median([row[45], row[46]])  # Median of Benefit and Happy
    negativity = np.median([row[47], row[48], row[49], row[50]])  # Median of Sad, Angry, Disgusted, Fearful
    word_ratings = pd.concat([word_ratings, pd.DataFrame({'word': [word], 'positivity': [positivity], 'negativity': [negativity]})], ignore_index=True)

print(word_ratings.head())  # Check the computed ratings

######################################################################
# Translate Chinese words to English using the provided translations file
######################################################################
translations_df = pd.read_csv(TRANSLATIONS_FILE_PATH, header=None)
translations_dict = dict(zip(translations_df.iloc[:, 0], translations_df.iloc[:, 1]))  # Create a dictionary for translation
word_ratings['word'] = word_ratings['word'].map(translations_dict)

print(word_ratings.head())  # Check the translated words

######################################################################
# Save the ratings to a CSV file for later use in RSA analyses
######################################################################
output_file = os.path.join(OUT_DIR, 'word_ratings.csv')
word_ratings.to_csv(output_file, index=False)
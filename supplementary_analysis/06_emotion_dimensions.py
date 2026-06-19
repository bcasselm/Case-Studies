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
from pathlib import Path
import numpy as np
import pandas as pd

######################################################################
# Configuration
######################################################################
PROJECT_ROOT = Path("/Volumes/T9/Birgit")
DATA_DIR = PROJECT_ROOT / "data"
BIDS_DIR = DATA_DIR / "ds004301"
RATINGS_FILE_PATH = BIDS_DIR / "derivatives" / "annotations" / "semantic feature" / "feature.csv"
TRANSLATIONS_FILE_PATH = BIDS_DIR / "derivatives" / "annotations" / "672words_translations.csv"
OUT_DIR = DATA_DIR / "emotion_ratings"
os.makedirs(OUT_DIR, exist_ok=True)

# The feature columns are labelled in Chinese. We select them by name (rather than by
# position) so the script is robust to column-order differences in the ratings file.
POSITIVE_FEATURES = ['益处', '快乐']            # Benefit, Happy
NEGATIVE_FEATURES = ['悲伤', '生气', '厌恶', '害怕']  # Sad, Angry, Disgusted, Fearful

######################################################################
# Data loading
######################################################################
ratings_df = pd.read_csv(RATINGS_FILE_PATH, encoding="utf-8-sig")

print(ratings_df.head())     # Check the structure of the ratings data
print(ratings_df.columns)    # Check column names to identify relevant features

######################################################################
# Compute positivity and negativity ratings for each word
######################################################################
word_ratings = pd.DataFrame()
for index, row in ratings_df.iterrows():
    word = row.iloc[0]  # The first column contains the word label
    positivity = np.median([row[c] for c in POSITIVE_FEATURES])  # Median of Benefit and Happy
    negativity = np.median([row[c] for c in NEGATIVE_FEATURES])  # Median of Sad, Angry, Disgusted, Fearful
    word_ratings = pd.concat([word_ratings, pd.DataFrame({'word': [word], 'positivity': [positivity], 'negativity': [negativity]})], ignore_index=True)

print(word_ratings.head())  # Check the computed ratings

######################################################################
# Translate Chinese words to English using the provided translations file
######################################################################
translations_df = pd.read_csv(TRANSLATIONS_FILE_PATH, header=None, encoding="utf-8-sig")
translations_dict = dict(zip(translations_df.iloc[:, 0], translations_df.iloc[:, 1]))  # Create a dictionary for translation
word_ratings['word_cn'] = word_ratings['word']   # preserve Chinese key before translation
word_ratings['word'] = word_ratings['word'].map(translations_dict)

print(word_ratings.head())  # Check the translated words

######################################################################
# Save the ratings to a CSV file for later use in RSA analyses
######################################################################
output_file = os.path.join(OUT_DIR, 'word_ratings.csv')
word_ratings[['word_cn', 'word', 'positivity', 'negativity']].to_csv(output_file, index=False)

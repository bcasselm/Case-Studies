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
# Helper functions
######################################################################
def load_ratings(ratings_file_path):
    """Loads feature ratings and prints column structure diagnostics."""
    ratings_df = pd.read_csv(ratings_file_path)

    print(ratings_df.head())  # Check the structure of the ratings data
    print(ratings_df.columns)  # Check column names to identify relevant features 
    # Columns are all in Chinese but the order of them corresponds to the order of the English translations in the authors paper, 
    # so we can identify the relevant columns based on their order (with first column being the word label):
    # Benefit (column 45), Happy (column 46), Sad (column 47), Angry (column 48), Disgusted (column 49), Fearful (column 50)
    
    return ratings_df


def compute_emotion_dimensions(ratings_df):
    """Calculates median positivity and negativity ratings for each word."""
    word_ratings = pd.DataFrame()
    for index, row in ratings_df.iterrows():
        word = row[0]  # Assuming the first column contains the word label
        positivity = np.median([row[45], row[46]])  # Median of Benefit and Happy
        negativity = np.median([row[47], row[48], row[49], row[50]])  # Median of Sad, Angry, Disgusted, Fearful
        word_ratings = pd.concat([word_ratings, pd.DataFrame({'word': [word], 'positivity': [positivity], 'negativity': [negativity]})], ignore_index=True)

    print(word_ratings.head())  # Check the computed ratings
    
    return word_ratings


def translate_words(word_ratings, translations_file_path):
    """Translates the Chinese word labels to English using the translations file."""
    translations_df = pd.read_csv(translations_file_path, header=None)
    translations_dict = dict(zip(translations_df.iloc[:, 0], translations_df.iloc[:, 1]))  # Create a dictionary for translation
    word_ratings['word'] = word_ratings['word'].map(translations_dict)

    print(word_ratings.head())  # Check the translated words
    
    return word_ratings


def save_ratings(word_ratings, out_dir):
    """Saves the final dataframe to a CSV file."""
    output_file = os.path.join(out_dir, 'word_ratings.csv')
    word_ratings.to_csv(output_file, index=False)
    print(f"Word ratings saved successfully to: {output_file}")


######################################################################
# Main execution
######################################################################
if __name__ == "__main__":
    # 1. Load the raw ratings 
    ratings_df = load_ratings(RATINGS_FILE_PATH)
    
    # 2. Compute the pos/neg dimensions 
    word_ratings = compute_emotion_dimensions(ratings_df)
    
    # 3. Translate Chinese labels to English 
    word_ratings_translated = translate_words(word_ratings, TRANSLATIONS_FILE_PATH)
    
    # 4. Save to CSV for later RSA analyses 
    save_ratings(word_ratings_translated, OUT_DIR)
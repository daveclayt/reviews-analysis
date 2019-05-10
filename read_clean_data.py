"""This script reads the csv file of scraped reviews from tripadvisor.com, reads
them into a dataframe and cleans the data."""

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from imblearn.under_sampling import RandomUnderSampler

ANALYZER = SentimentIntensityAnalyzer()

FILE = 'scrapy_output.csv'

TAKENUMS = r'\d+'
REMCHEV = r'\<(.*?)\>'
REMSPACE = '^[ \t]+|[ \t]+$'
FIRST3CHARS = '^...'
NUMSONLY = '[^0-9]'

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# list of "stop words" I wanted to keep as they carry semantic importance
STOP_WORDS.remove("not")
STOP_WORDS.remove("but")
STOP_WORDS.remove("only")
STOP_WORDS.remove("against")
STOP_WORDS.remove("again")
STOP_WORDS.remove("nor")
STOP_WORDS.remove("no")
STOP_WORDS.remove("off")

def read_data(path):
    """reads in a csv file from a defined path and drops duplicates"""
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    # select and delete an erroneous row that appears in the scraped csv file
    delete = df.index[df['restaurant_rating'] == 'restaurant_rating'][0]
    df.drop(delete, inplace=True)

    return df

def get_review_ratings(string1):
    """takes the required numbers from the review ratings column"""
    return int(re.findall(TAKENUMS, string1)[0]) / 10

def remove_chevron_contents(string1):
    """removes unwanted html tags from reviews"""
    return re.sub(REMCHEV, '', string1)

def remove_whitespace(string1):
    """removes unwanted whitespace from start and end of restaurant names"""
    return re.sub(REMSPACE, '', string1)

def get_restaurant_ratings(string1):
    """extracts restaurant ratings from strings"""
    return float(re.findall(FIRST3CHARS, string1)[0])

def remove_non_nums(string1):
    """extracts numeric characters from a string"""
    return int(re.sub(NUMSONLY, '', string1))

def get_vader_scores(review):
    """returns compound score from VADER for each sentence"""
    scores = []
    for sentence in review:
        scores.append(ANALYZER.polarity_scores(sentence)['compound'])
    return scores

def get_tokens(review):
    """tokenize each review into words"""
    # repmove full stops (nltk does not separate words with stop and no space)
    nodots = re.sub(r'\.', ' ', review)
    # split review into list of word 'tokens'
    tokens = word_tokenize(nodots)
    # convert to lower case
    lowercase = [token.lower() for token in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in lowercase]
    # filter out remaining non-alphabetic tokens
    words = [word for word in stripped if word.isalpha()]
    # finally, remove stop words
    result = [w for w in words if not w in STOP_WORDS]
    return result


# regular expressions to clean the text:


# apply functions to clean our data
def clean_dataframe(df):
    """apply helper functions to clean our dataframe"""
    df['review_rating'] = df['review_rating'].apply(get_review_ratings)
    df['full_reviews'] = df['full_reviews'].apply(remove_chevron_contents)
    df['restaurant_name'] = df['restaurant_name'].apply(remove_whitespace)
    df['restaurant_rating'] = df['restaurant_rating'].apply(get_restaurant_ratings)
    df['restaurant_review_count'] = df['restaurant_review_count'].apply(remove_non_nums)

    # combine title and review in order to include title in sentiment analysis
    df['title_plus_review'] = df['review_title'].astype(str) + ". " + df['full_reviews'].astype(str)

    ### First we run VADER to get some sentiment analysis
    # df['sentences'] = df['title_plus_review'].apply(sent_tokenize)
    # df['vader_scores'] = df['sentences'].apply(get_vader_scores)
    # df['avg_vader_score'] = df['vader_scores'].apply(np.mean)

    df['words'] = df['full_reviews'].apply(get_tokens)
    df['word_count'] = df['words'].apply(len)

    return df

def prep_model_inputs(df):
    """vectorizes words usine pre-trained gensim vectorizer, pads reviews to
    30 words, performs train-test-split and formats the data ready for model input"""
    wv = gensim.models.word2vec.Word2Vec(df['words'], size=15, window=5, min_count=1)
    vocabulary = list(wv.wv.vocab)
    d = dict(zip(vocabulary, range(len(vocabulary))))
    df['seq'] = [[d[word] for word in review] for review in df['words']]
    padded = pad_sequences(df['seq'], maxlen=30)
    X = padded
    y = df['review_rating']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, random_state=42)

    rus = RandomUnderSampler(random_state=42) # downsample to fix class imbalance
    X_res, y_res = rus.fit_resample(Xtrain, ytrain)
    X_res, y_res = shuffle(X_res, y_res)
    y_res_1h = pd.get_dummies(y_res)

    # get imbedding weights
    weights = []
    for word in vocabulary:
        weights.append(wv[word])

    embedding_weights = np.array(weights)
    return vocabulary, embedding_weights, X_res, y_res_1h


def main(path):
    """runs the above functions"""
    print('reading data...')
    df = read_data(path)
    print('cleaning data...')
    df_clean = clean_dataframe(df)
    print('preparing model inputs...')
    vocab, ew, X, y = prep_model_inputs(df_clean)
    return vocab, ew, X, y


if __name__ == '__main__':
    vocab, ew, X, y = main(FILE)
    print(len(vocab))

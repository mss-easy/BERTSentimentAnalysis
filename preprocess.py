import re 
import nltk
from nltk.corpus import stopwords
from string import punctuation

def preprocess_text(tweet):
    tweet = tweet.lower()
    # replace links with 'url'
    tweet = re.sub(r'((https?:\/\/)|(www\.))[A-Za-z0-9.\/]+', 'url',  tweet)
    tweet = re.sub(r'[A-Za-z0-9]+.com', 'url',tweet)
    # remove , @users, if any
    tweet = re.sub(r'[@][A-Za-z0-9]+', '',tweet)
    # remove non-ascii characters
    tweet = ''.join([w for w in tweet if ord(w)<128])
    #get hastags
    """
    # first i thought of processing hastags in other field than sentence because sometime hastags itself carry sentiment
    # bert tokenizer takes care of such tokens, which have some punctuation attached to it
    # hastags will be taken care of 
    tags = ' '.join([w.strip("#") for w in tweet.split() if w.startswith("#")])
    tweet = re.sub(r'[#][A-Za-z0-9]+', '',tweet)
    """
    tweet = tweet.strip()
    # return tweet, tags
    return tweet
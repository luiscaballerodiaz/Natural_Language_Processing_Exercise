from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


class LemmaTokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        regex_num_punctuation = r'(\d+)|([^\w\s])'
        regex_little_words = r'(\b\w{1,2}\b)'
        regex_countvectorizer = r'(?u)\b\w\w+\b'
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if re.search(regex_countvectorizer, t)]

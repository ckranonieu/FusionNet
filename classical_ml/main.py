import numpy as np
import pandas as pd
import re
import nltk
import string

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier

from bs4 import BeautifulSoup

from nltk.stem import WordNetLemmatizer 

'''
Linguistic Preprocessing: Refurbished code taken from what was presented in class. 
Courtesy of ChatGPT. 

Main suggestion was to compile regular expressions
before any code is run in order to decrease look-up time. (According to output.)

All else, upon visual examination, appears to be the same.
'''

class LinguisticPreprocessor(BaseEstimator, TransformerMixin):
    def _download_if_non_existent(self, res_path, res_name):
        try:
            nltk.data.find(res_path)
        except LookupError:
            # print(f'resource {res_path} not found. Downloading now...')
            nltk.download(res_name, quiet=True)
    
    
    def __init__(self, do_lemmatize=False, remove_html=False):
        self.do_lemmatize = do_lemmatize
        self.remove_html = remove_html
        
        if self.do_lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            nltk.download('punkt', quiet=True)

        nltk.download('omw-1.4', quiet=True)
        nltk.download("wordnet", quiet=True)

        self._download_if_non_existent('corpora/stopwords', 'stopwords')
        self._download_if_non_existent('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        self._download_if_non_existent('corpora/wordnet', 'wordnet')

        # Compile regex patterns for efficiency
        self.punct_and_num_pattern = re.compile('[%s\d]' % re.escape(string.punctuation))
        self.double_space_pattern = re.compile(" +")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed_X = []
        for text in X:
            processed_text = text
            if self.remove_html:
                processed_text = self._remove_html_tags(processed_text)
            processed_text = self._process_text(processed_text)
            if self.do_lemmatize:
                processed_text = self._lemmatize_text(processed_text)
            transformed_X.append(processed_text)
        return transformed_X

    def _remove_html_tags(self, text):
        return BeautifulSoup(text, 'html.parser').get_text()

    def _process_text(self, text):
        text = text

        # Remove punctuations and numbers if specified
        text = self.punct_and_num_pattern.sub('', text)
        # Remove double spaces
        text = self.double_space_pattern.sub(' ', text)
        return text

    def _lemmatize_text(self, text):
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

def fetch_data(use_imdb = True):
    train = pd.read_parquet('rotten_tomatoes_train')
    test = pd.read_parquet('rotten_tomatoes_test')
    val = pd.read_parquet('rotten_tomatoes_val')
    
    # Inclusion of validation dataset (Cross-Validation was used.)
    X_train = train['text']._append(val['text']).reset_index(drop=True)
    y_train = train['label']._append(val['label']).reset_index(drop=True)
    
    # Loads imdb dataset.
    imdb_train = pd.read_parquet('imdb_train')
    imdb_test = pd.read_parquet('imdb_test')
        
    # Combines the two partitions from imdb dataset together.
    imdb_X_train = imdb_train['text']._append(imdb_test['text']).reset_index(drop=True)
    imdb_y_train = imdb_train['label']._append(imdb_test['label']).reset_index(drop=True)

    # Folds combined imdb dataset into training set. 
    X_train = X_train._append(imdb_X_train).reset_index(drop=True)
    y_train = y_train._append(imdb_y_train).reset_index(drop=True)

    # Test dataset kept separate.
    X_test =  test['text']
    y_test =  test['label']

    return (X_train, X_test, y_train, y_test)



'''
Intalizes a 2 MultinomialNB + 1 SGD Classifier Weighted Voting Ensemble,
where optimal weights were searched for using Optuna.
'''

class CustomEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha_nb1, alpha_nb2, alpha_sgd, weight_nb1, weight_nb2, weight_sgd):
        self.alpha_nb1 = alpha_nb1
        self.alpha_nb2 = alpha_nb2
        self.alpha_sgd = alpha_sgd
        self.weight_nb1 = weight_nb1
        self.weight_nb2 = weight_nb2
        self.weight_sgd = weight_sgd
        self.nb1 = MultinomialNB(alpha=self.alpha_nb1)
        self.nb2 = MultinomialNB(alpha=self.alpha_nb2)
        self.sgd = SGDClassifier(alpha=self.alpha_sgd, loss='log_loss')

    def fit(self, X, y):
        self.nb1.fit(X, y)
        self.nb2.fit(X, y)
        self.sgd.fit(X, y)
        return self

    def predict(self, X):
        nb1_pred = self.nb1.predict_proba(X) * self.weight_nb1
        nb2_pred = self.nb2.predict_proba(X) * self.weight_nb2
        sgd_pred = self.sgd.predict_proba(X) * self.weight_sgd

        # Sum and then normalize to get weighted average
        total_weight = self.weight_nb1 + self.weight_nb2 + self.weight_sgd
        avg_pred = (nb1_pred + nb2_pred + sgd_pred) / total_weight
        return np.argmax(avg_pred, axis=1)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = fetch_data(use_imdb = True)
    
    '''
    
    Three step pipeline 
    
      Preprocessing step:
          Implements preprocessing steps shown in class.
          In addition, incorporates imdb dataset from HuggingFace and folds Validation set into train set. 

    '''
    pipeline_steps = [('preprocessor', LinguisticPreprocessor(
                        do_lemmatize=False,
                        remove_html=False)),
                      ('vectorizer', TfidfVectorizer(
                                       max_df=0.9689718322672588,
                                       min_df=2,
                                       ngram_range= (1, 3))),
                      ('classifier', CustomEnsemble(
                                        alpha_nb1 = 0.01637039915924717,
                                        alpha_nb2 = 0.04387811599354438,
                                        alpha_sgd = 1.3858253028497528e-06,
                                        weight_nb1 = 0.1199906867752188,
                                        weight_nb2 = 0.12449360259633686,
                                        weight_sgd = 0.9497473406190401
                          

                      ))]

    pipeline = Pipeline(steps=pipeline_steps)

    pipeline.fit(X_train, y_train)

    # Predict on the test data
    y_pred = pipeline.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))

    data_dict = {'index': np.arange(len(y_pred)),
                 'pred': y_pred}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    # Write the DataFrame to a CSV file without the DataFrame's index
    df.to_csv('results.csv', index=False)

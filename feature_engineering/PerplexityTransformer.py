import transformers
import evaluate
from evaluate import load

import nltk
nltk.download('punkt')

import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class PerplexityTransformer(BaseEstimator, TransformerMixin):
    # X should be a dataframe and it should have a response column
    def __init__(self):
        self.perplexity_scraper = PerplexityScraper()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.perplexity_scraper.get_perplexity(X)


class PerplexityScraper():
    def __init__(self):
        self.perplexity = load("perplexity", module_type="metric")

    def get_perplexity(self, df):
        feature_arrays = []
        for index, row in df.iterrows():
            text = row["response"]
            mean_perplexity = self.calculate_perplexity_of_text(text)
            row_values = df.loc[index, :].values.flatten().tolist()
            row_values.append(mean_perplexity)
            feature_arrays.append(row_values)

        new_columns = df.columns.to_list()
        new_columns.extend(['Mean Perplexity'])
        perplexity_df = pd.DataFrame(feature_arrays, columns=new_columns)
        return perplexity_df

    def calculate_perplexity_of_text(self, text):
        input_texts = nltk.sent_tokenize(text)
        input_texts = [x for x in input_texts if x != '' and len(x) > 2]
        results = self.perplexity.compute(model_id='gpt2', add_start_token=True, predictions=input_texts)
        mean_perplexities = results["mean_perplexity"]
        return mean_perplexities
import language_tool_python
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
from textblob import TextBlob
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class LanguageFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')
        self.language_feature_scraper = LanguageFeatureScraper()

    def fit(self, X, y=None):
        return self

    # X should be a dataframe and it should have a response column
    def transform(self, X, y=None):
        return self.language_feature_scraper.get_language_features(X,self.tool)

class LanguageFeatureScraper():
    def __init__(self):
        pass

    def get_language_features(self, df_train,tool):
        final_col_lst = df_train.columns.to_list()
        final_col_lst.append("sum_mistakes")
        mistakes = self.get_mistakes(df_train,tool)
        mistakes_unique = self.get_unique_list(mistakes)
        df_processed = self.get_features(mistakes, mistakes_unique, df_train)
        df_processed = df_processed[final_col_lst]
        df_processed["TextBlob_Subjectivity"] = df_processed["response"].apply(self.getSubjectivity)
        df_processed["Formality Score"] = df_processed["response"].apply(self.calculate_formality_score)
        return df_processed

    def get_mistakes(self, df, tool):
        mistakes = []
        for text in df['response'].values:
            mistakes.append(tool.check(text))
        return mistakes

    def get_unique_list(self, mistakes):  # not for submission
        mistake_list = [item for sublist in
                        [[mistakes[i][j].ruleIssueType for j in range(len(mistakes[i]))] for i in range(len(mistakes))]
                        for item in sublist]
        mistakes_unique = [item for item in set(mistake_list)]
        return mistakes_unique

    def get_features(self, mistakes, mistakes_unique, df):
        df[mistakes_unique] = 0
        df['error_length'] = 0
        idx = 0
        for item in mistakes:
            if len(item) > 0:
                for i in range(len(item)):
                    if item[i].ruleIssueType == 'grammar':
                        df.iloc[idx, df.columns.get_loc('grammar')] += 1
                    elif item[i].ruleIssueType in mistakes_unique and item[i].ruleIssueType != 'grammar':
                        df.iloc[idx, df.columns.get_loc(item[i].ruleIssueType)] += 1
                    df.iloc[idx, df.columns.get_loc('error_length')] += item[i].errorLength
            idx += 1
        df['sum_mistakes'] = df[mistakes_unique].sum(axis=1)
        return df

    def getSubjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    def conduct_spacy_pos_tag(self, txt):
        doc = nlp(txt)
        pos_list = []
        for token in doc:
            pos_list.append(token.pos_)
        return pos_list

    def calculate_formality_score(self, txt):
        pos_tagged = self.conduct_spacy_pos_tag(txt)
        count = Counter(pos_tagged)

        noun = count["NOUN"]
        adj = count["ADJ"]
        prep = count["ADP"]
        article = count["DET"]
        pronoun = count["PRON"]
        verb = count["VERB"]
        adverb = count["ADV"]
        interjection = count["INTJ"]
        conj = count["CONJ"]

        N = noun + adj + prep + article + pronoun + adverb + interjection + conj
        fc = noun + adj + prep + article
        nc = pronoun + verb + adverb + interjection
        if N == 0:
            return 0
        return 50 * ((fc - nc) / N + 1)


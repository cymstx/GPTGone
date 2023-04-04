"""
For RNN Feature Classifier
!pip install tensorflow_text
"""
import autokeras as ak

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
from keras.models import load_model


def flatten(l):
    return [item for sublist in l for item in sublist]


def process_output(preds):
    flattened_pred = flatten(preds)
    output = tf.constant(flattened_pred)
    return tf.math.sigmoid(output).numpy().tolist()


class RNNPerplexityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, path):
        self.model = load_model(path)

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return process_output(self.model.predict(processed_X))

    # Add any custom transformers as needed to produce the X input format your model needs
    def process(self, X, y=None):
        output = [X[["Mean Perplexity"]].values, X["response"].values]
        return output


class CNNPerplexityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, path):
        self.model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return process_output(self.model.predict(processed_X))

    # Add any custom transformers as needed to produce the X input format your model needs
    def process(self, X, y=None):
        output = [X[["Mean Perplexity"]], X["response"].to_numpy()]
        return output


class CNNGLTRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, path):
        self.model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return process_output(self.model.predict(processed_X))

    # Add any custom transformers as needed to produce the X input format your model needs
    def process(self, X, y=None):
        output = [X[["GLTR Category 1", "GLTR Category 3"]],
                  X["response"].to_numpy()]
        return output


class CNNTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, path):
        self.model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return process_output(self.model.predict(processed_X))

    # Add any custom transformers as needed to produce the X input format your model needs
    def process(self, X, y=None):
        output = X["response"].to_numpy()
        return output

import numpy as np
import pandas as pd

from classifiers.BertBaseClassifier import BertBaseClassifier
from classifiers.SvmTfidfBaseClassifier import SvmTfidfBaseClassifier
from classifiers.KerasModelClassifier import CNNGLTRClassifier
from Ensemble import Ensemble
from sklearn.pipeline import Pipeline
from model_development.feature_engineering.GLTRTransformer import GLTRTransformer

def predict(text):
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])

    pipeline = Pipeline(steps=[('GLTR', GLTRTransformer())])
    processed_input_df = pipeline.fit_transform(sample_df)

    bert = BertBaseClassifier("model_bertbase_updated.pt")
    svm = SvmTfidfBaseClassifier("svm_rbf_model_no_gltr.pkl", "tfidf_vectorizer.pkl")
    cnn = CNNGLTRClassifier("model_autokeras_gltr")

    #models = [bert, svm_tfidf_classifier]
    models=[bert, cnn, svm]
    ensemble = Ensemble(models, ["BERT","CNN","SVM"])

    #pred is an output "AI" or "Human" and output_dict shows what prediction each model made
    threshold = 0.6
    weights = np.array([0.25, 0.25,0.5])
    pred,  output_dict = ensemble.predict(processed_input_df, weights, threshold)

    print(pred)
    print(output_dict)

ai_text = "Block modeling is like building a big puzzle! Imagine you have lots of puzzle pieces, but they are all different shapes and colors. Block modeling helps you group together pieces that are similar, so you can see how they fit together.In block modeling, you have a big picture with lots of dots and lines. Each dot is a person, and each line shows if two people are friends. But some people might have more friends than others, or might be friends with different kinds of people. So, block modeling helps you group together people who are similar, based on things like their age, gender, or hobbies.Once you group people together, you can see how the groups are connected to each other. It's like putting together puzzle pieces that are the same color, so you can see how they fit into the bigger picture. By using block modeling, you can understand how people are connected in the big picture of the social network, and how the groups of people are related to each other."
predict(ai_text)
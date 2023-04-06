from sklearn.pipeline import Pipeline

from GLTRTransformer import GLTRTransformer
from PerplexityTransformer import PerplexityTransformer
from LanguageFeatureTransformer import LanguageFeatureTransformer

import pandas as pd
df = pd.read_csv("data/holdout.csv")
scraping = Pipeline(steps=[('GLTR', GLTRTransformer()), ('Perplexity',PerplexityTransformer()), ("Language features",LanguageFeatureTransformer())])
try:
    print("Starting to scrape")
    df_final = scraping.fit_transform(df)
    df_final.to_csv("holdout_with_features.csv",index=False)
except Exception as e:
    df_final.to_csv("holdout_with_features.csv",index=False)
    print("Error", e)

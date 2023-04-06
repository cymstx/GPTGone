import transformers
import evaluate
from evaluate import load
import nltk
nltk.download('punkt')
import torch
import numpy as np
import pandas as pd
import csv
import json

perplexity = load("perplexity", module_type="metric")

def calculate_perplexity_of_text(text):
  input_texts = nltk.sent_tokenize(text)
  input_texts = [x for x in input_texts if x != '' and len(x) > 2]
  results = perplexity.compute(model_id='gpt2',add_start_token=True, predictions=input_texts)
  mean_perplexities = results["mean_perplexity"]
  return mean_perplexities


raw_filename = "eli5_clean_combined.csv"
compiled_filename = "Perplexity_Compiled.csv"
raw_df = pd.read_csv(raw_filename)
compiled_perplexity_df = pd.read_csv(compiled_filename)

start_index = len(compiled_perplexity_df)
batch_size = 50
end_index = start_index + batch_size

while end_index != len(raw_df)-1:
  print(f"Starting batch with start index = {start_index} and end_index = {end_index}")
  df = raw_df.iloc[start_index:end_index]
  feature_arrays = []
  for index, row in df.iterrows():
    text = row["response"]
    mean_perplexity = calculate_perplexity_of_text(text)
    row_values = df.loc[index, :].values.flatten().tolist()
    row_values.append(mean_perplexity)
    feature_arrays.append(row_values)

  print(f"Writing to CSV for {start_index} to {end_index}")
  new_columns = df.columns.to_list()
  new_columns.extend(['Mean Perplexity'])

  perplexity_df = pd.DataFrame(feature_arrays, columns = new_columns)
  compiled_perplexity_df = pd.read_csv(compiled_filename)
  compiled_perplexity_df = pd.concat([compiled_perplexity_df, perplexity_df])
  compiled_perplexity_df.to_csv(compiled_filename, index=False)

  start_index = end_index
  end_index = start_index + batch_size

"""
raw_df = pd.read_csv("eli5_clean_combined.csv")
headers = raw_df.columns.to_list()
headers.extend(['Mean Perplexity'])
print(headers)
filename = 'Perplexity_Compiled.csv'

with open(filename, 'w', newline='') as csvfile:
   csvwriter = csv.writer(csvfile)
   csvwriter.writerow(headers)
   csvfile.close()
"""
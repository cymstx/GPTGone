from chatgpt_wrapper import ChatGPT
import pandas as pd
import openai

# Set up the OpenAI API client
openai.api_key = ""

# load the dataset
df = pd.read_csv("datasets/Dataset0.csv")
print(df.head())

# add a column to the dataset if there are less than 3 columns
if len(df.columns) < 3:
  df["chatgpt_response"] = None

# set the saving frequency of the dataset
saving_frequency = 20

# choose the first row with a blank entry in column 3 of df
question_index = df[df.chatgpt_response.isnull()].index[0]
while(question_index <= 3272):
    # select the question from the dataset
    prompt = df.iloc[question_index, 1]
    custom_msg = [
      {"role": "user", "content": prompt}
    ]
    try:
      # Generate a response
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=custom_msg
      )
      response_str = response['choices'][0]["message"]["content"]

      # add the response to the dataset
      df.iloc[question_index, 2] = response_str.strip()

      print(f"Question: {prompt[:100]}")
      print(f"Response: {response_str[:100]}")

      # check if the saving frequency has been reached
      if question_index % saving_frequency == 0:
        df.to_csv("datasets/Dataset0.csv", index=False)
        print(f"\nDataset saved till row {question_index}\n")

      question_index += 1

    except Exception as e:
      print("Error asking ChatGPT", e)
      df.to_csv("datasets/Dataset0.csv", index=False)
      break

# save the dataset
df.to_csv("datasets/Dataset0.csv", index=False)
print(f"\nDataset saved till row {question_index}\n")
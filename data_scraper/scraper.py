from chatgpt_wrapper import ChatGPT
import pandas as pd
import signal
import time
import sys

def timeout_handler(signum, frame):
  print("request timeout")
  raise Exception("Timed out!")

dataset_path = "datasets/Dataset1.csv"

# load the dataset
df = pd.read_csv(dataset_path)

# add a column to the dataset if there are less than 3 columns
if len(df.columns) < 3:
  df["chatgpt_response"] = None

# set the number of failures before terminating the script
threshold = 5
number_of_failures = 0

# set the saving frequency of the dataset
saving_frequency = 20

# choose the first row with a blank entry in column 3 of df
question_index = df[df.chatgpt_response.isnull()].index[0]

bot = ChatGPT()

# register the signal function handler
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 120

while(number_of_failures< threshold):
  try:
    # set the timeout for 60 seconds
    signal.alarm(timeout)

    # select the question from the dataset
    question = df.iloc[question_index, 1]
    print(f"Question {question_index}: {question[:100]}")

    response = bot.ask(question)
    if response == "Unusable response produced, maybe login session expired. Try 'pkill firefox' and 'chatgpt install'":
      # wait for 45 minutes before trying again
      print("Sleeping for 45 minutes")
      df.to_csv(dataset_path, index=False)
      print(f"\nDataset saved till row {question_index}\n")
      time.sleep(40*60)
      raise Exception("Unusable response produced, maybe login session expired. Try 'pkill firefox' and 'chatgpt install'")

    # add the response to the dataset
    df.iloc[question_index, 2] = response.strip()

    # check if the saving frequency has been reached
    if question_index % saving_frequency == 0:
      df.to_csv(dataset_path, index=False)
      print(f"\nDataset saved till row {question_index}\n")
    
    print(f"Response: {response[:100]}")

    # cancel the timeout
    signal.alarm(0)

    number_of_failures=0
    question_index += 1
  
  except Exception as e:
    print(e)
    number_of_failures += 1
    # referesh chatgpt session
    bot.refresh_session()

# save the dataset
df.to_csv(dataset_path, index=False)
print(f"\nDataset saved till row {question_index}\n")
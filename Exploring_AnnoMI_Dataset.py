import pandas as pd
import re # importing regular expression library

df = pd.read_csv("AnnoMI_dataset.csv") # read the dataset
print(df.iloc()[0])

# Data Processing
# code below is to construct an array that groups instruction with input and response as an element. Thus extracting the conversation

transcript_id = 0

final_list = []

for index, row in df.iterrows():

  if row["interlocutor"] == "client":
    row_transcript_id = row.transcript_id
    interlocutor = row["interlocutor"]
    client_utterance = row.utterance_text

    response_row = df.iloc()[index+1]
    response_utterance = response_row.utterance_text

    client_utterance = re.sub(r'\[.*?\]', '', client_utterance)
    response_utterance = re.sub(r'\[.*?\]', '', response_utterance)


    final_list.append(
        {
         "instruction" : client_utterance,
         "input" : "",
         "output": response_utterance
         }
    )
# Convert data to json format to be processed by ALPACA algorithm later
import json

# Define a list to save to a JSON file
my_list = final_list

# Define the file path and name
file_name = "counseling_data.json"

# Open the file for writing and save the list to it
with open(file_name, "w") as json_file:
    json.dump(my_list, json_file)

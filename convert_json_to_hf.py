# to leverage the HuggingFace transformers library we will need to transform this json file to 
# a format that HuggingFace can read
# the below code will do just that
# Convert JSON dataset to HF format
import json
from datasets import Dataset, DatasetDict # this dependency is required !pip install datasets

# We also would like this HF dataset to have a specific output
# the output desired is 3 columns for instruction, text and target.
def preprocess_data(dataset_path: str):
    raw_data = json.load(open("counselling_data.json"))
    instructions = []
    inputs = []
    outputs = []

    for data in raw_data:
        instructions.append(data["instruction"])
        inputs.append(data["input"])
        outputs.append(data["output"])

    data_dict = {
        "train": {"instruction": instructions, "text": inputs, "target": outputs}
    }

    dataset = DatasetDict()
    # using your `Dict` object
    for k, v in data_dict.items():
        dataset[k] = Dataset.from_dict(v)

    dataset.save_to_disk(str("counseling_data_HF"))

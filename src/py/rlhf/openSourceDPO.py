from transformers import AutoTokenizer
from datasets import load_dataset
 
# Load Tokenizer from the hub
model_id = "cognitivecomputations/dolphin-2.1-mistral-7b" # replace with your model id
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Load dataset from the hub
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
dataset = dataset.shuffle().select(range(13750))
 
 
def rec_extract_assistant_messages(messages, index=-1):
  """Recursively extract the last assistant messages from the end of the conversation."""
  if messages[index]["role"] == "assistant":
    return [messages[index]]
  else:
    return rec_extract_assistant_messages(messages, index-1)
 
# System message used if there is no system message at the beginning of the conversation
# Can be repelaced and modified as needed
DEFAULT_SYSTEM_MESSAGE = "You are Dolphin, a helpful AI assistant."
 
def create_triplets(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
  """Create the triplets (prompt, chosen, rejected)"""
  # Extract the N-1 turns to form the prompt
  # Prepend a system message if the first message is not a system message
  prompt_messages = example["chosen"][:-1]
  if example["chosen"][0]["role"] != "system":
      prompt_messages.insert(0, {"role": "system", "content": default_system_message})
  # Now we extract the final assistant turn to define chosen/rejected responses
  chosen_messages = rec_extract_assistant_messages(example["chosen"])
  rejected_messages = rec_extract_assistant_messages(example["rejected"])
 
  # apply template to the messages and return the triplets
  returnTriplets = {
    "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
    "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
    "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False)
  }
  return returnTriplets
 
dataset = dataset.map(create_triplets, remove_columns=dataset.features, fn_kwargs={"tokenizer": tokenizer})
# split dataset into 11,000 training samples and 2,750 test samples
dataset = dataset.train_test_split(test_size=2750/13750)
 
# print sample cut of
print(dataset["train"][0]["prompt"][:50])
print(dataset["train"][0]["chosen"][:50])
print(dataset["train"][0]["rejected"][:50])
 
# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")
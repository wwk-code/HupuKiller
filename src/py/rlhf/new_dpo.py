import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer
import bitsandbytes as bnb


# Define model names and tokens
peft_model_name = "Ronal999/phi2_finance_SFT" # The model obtained after the SFT step
new_model = "phi2_DPO" #the name of the DPO trained model

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Helper function to format the dataset
def chatml_format(example):
    # Formatting system response
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Formatting user instruction
    message = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Formatting the chosen answer
    chosen = example['chosen'] + "\n"

    # Formatting the rejected answer
    rejected = example['rejected'] + "\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Loading the dataset
dataset = load_dataset("Intel/orca_dpo_pairs")['train']

# Saving original columns for removal
original_columns = dataset.column_names

# Applying formatting to the dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)

# Displaying a sample from the dataset
print(dataset[1])

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj', 'dense']
)

# Load the base model with BitsAndBytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    is_trainable=True,
)

model.config.use_cache = False
model.load_adapter(peft_model_name, adapter_name="training2")
model.load_adapter(peft_model_name, adapter_name="reference")

# Initialize Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    max_steps=50, # we set up the max_steps to 50, due to free GPU useage
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=5,
    remove_unused_columns=False,
)

# Initialize DPO Trainer
dpo_trainer = DPOTrainer(
    model,
    model_adapter_name="training2",
    ref_adapter_name="reference",
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1, # The parameter 'beta' is the hyperparameter of the implicit reward and is normally set from 0.1 to 0.5. It's important to note that if beta tends to zero, we tend to ignore the reference model.
    max_prompt_length=512,
    max_length=1024,
)


# Start Fine-tuning with DPO
dpo_trainer.train()

# Saving the fine-tuned model and tokenizer
dpo_trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

# Releasing memory resources
del dpo_trainer, model
gc.collect()
torch.cuda.empty_cache()

# Loading the base model and tokenizer
base_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    return_dict=True
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

# Merging the base model with the adapter and unloading
model = PeftModel.from_pretrained(base_model, "final_checkpoint")
model = model.merge_and_unload()

# Saving the merged model and tokenizer
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

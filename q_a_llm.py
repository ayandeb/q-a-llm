# Q&A


# pip install transformers peft datasets accelerate bitsandbytes

from huggingface_hub import hf_hub_download, login
import torch
import os
import json
import transformers
from transformers import pipeline
from huggingface_hub import hf_hub_download, login
from transformers import AutoModel, AutoTokenizer, LlamaConfig


os.environ['HF_TOKEN'] = "***"  #secret key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "***"

# Load the LLAMA 3.1 Model and QLoRA:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Log in using the token from environment variables
huggingface_token = os.environ.get('HF_TOKEN')
login(token=huggingface_token)

# Define model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Download the configuration file
config_path = hf_hub_download(repo_id=model_name, filename="config.json", use_auth_token=True)

# Load and adjust the configuration
with open(config_path, 'r') as f:
    config_data = json.load(f)

# Modify the `rope_scaling` field
if 'rope_scaling' in config_data and isinstance(config_data['rope_scaling'], dict):
    config_data['rope_scaling'] = {
        'type': 'linear',  # Replace 'default' with the correct type if known
        'factor': config_data['rope_scaling'].get('factor', 1.0)
    }

# Save the modified configuration locally
modified_config_path = "modified_config.json"
with open(modified_config_path, 'w') as f:
    json.dump(config_data, f)

# Load the modified configuration
config = LlamaConfig.from_json_file(modified_config_path)

# Load the tokenizer and model with the modified configuration
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModel.from_pretrained(model_name, config=config, use_auth_token=True)


# Setup QLoRA with PEFT:
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Specific layers to apply LoRA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Prepare the Dataset:
# Load your dataset using Hugging Face datasets.
from datasets import load_dataset

dataset = load_dataset("squad")  # Example: SQuAD dataset

# Tokenize the dataset for training:
def preprocess_function(examples):
    return tokenizer(examples['question'], examples['context'], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Fine-Tune the Model:
# Use Hugging Face’s Trainer for training the model:
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()

# Save the Fine-Tuned Model:
model.save_pretrained("llama_finetuned_peft_qlora")
tokenizer.save_pretrained("llama_finetuned_peft_qlora")



###
# If I run same code again, will it again download? will overwrite the previously downloaded model?or will download in separate space and consume more memeory?
# ChatGPT said:
# ChatGPT
# When you run the same code again, it will not re-download the model as long as it is already downloaded and cached on your system. Hugging Face uses a caching mechanism to store models locally. Here's what happens when you run the code again:

# 1. Cached Model:
# Hugging Face stores downloaded models in a cache directory (e.g., ~/.cache/huggingface/transformers/ or C:\Users\<YourUsername>\.cache\huggingface\).
# When you run the code again, the library checks if the model is already present in the cache. If it is, it will load the model from there, and no re-download will occur.
# This prevents the same model from being downloaded multiple times, thus saving bandwidth and disk space.
# 2. No Overwrite:
###

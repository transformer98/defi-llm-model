from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login
import os

MAX_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5

examples = [
    {
        "input": "I want to swap 0.5 ETH to USDC on Polygon",
        "output": '''```json\n{
  "inputToken": "ETH",
  "outputToken": "USDC",
  "amount": "0.5",
  "chain": "polygon",
  "slippage": 0.5
}\n```'''
    },
    {
        "input": "Swap 300 USDT from Ethereum to Arbitrum",
        "output": '''```json\n{
  "inputToken": "USDT",
  "outputToken": "USDT",
  "amount": "300",
  "fromChain": "ethereum",
  "toChain": "arbitrum",
  "slippage": 0.5
}\n```'''
    }
]

# Login with your token (store in env var or hardcoded securely)
HF_TOKEN = os.getenv("HF_TOKEN") or "hf_your_access_token_here"
login(token=HF_TOKEN)

# model name or path
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" # model name
MODEL_NAME = "./models/mistral-7b-instruct" #downloaded model path
# MODEL_NAME = "google/gemma-2b-it" #model name
OUTPUT_DIR = "./my-swap-model"

dataset = Dataset.from_list(examples)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

def preprocess(example):
    text = example["input"] + "\n\n" + example["output"]
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_strategy="epoch",
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

print("Saving trained model in .safetensors format...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved at {OUTPUT_DIR}")


### ************* test model ************* ###

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Define a prompt for testing
test_input = "Swap 150 DAI from Binance Smart Chain to Ethereum"

# Tokenize the input
inputs = tokenizer(test_input, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)

# Generate output
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=MAX_LENGTH,
    num_beams=4,
    early_stopping=True
)

# Decode the output
generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nTest Prompt:")
print(test_input)
print("\nGenerated Output:")
print(generated_output)
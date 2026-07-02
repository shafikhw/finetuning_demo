import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "natural-disaster-lora"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    max_length=512,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

print(f"Saved LoRA adapter to: {OUTPUT_DIR}")
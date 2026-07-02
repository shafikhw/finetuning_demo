import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_PATH = "natural-disaster-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a disaster response assistant. Classify the natural disaster report and reply in strict JSON with keys: disaster_type, severity, recommended_action."
    },
    {
        "role": "user",
        "content": "The river overflowed after nonstop rain, and nearby houses are surrounded by water."
    }
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    input_text,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(response)
# Natural Disaster LLM Fine-Tuning Demo

A simple demo showing how to fine-tune a small language model to classify natural disaster reports and return a structured JSON response.

## Task

The model takes a disaster report as input and returns:

```json
{
  "disaster_type": "flood",
  "severity": "high",
  "recommended_action": "Move to higher ground if safe, avoid floodwater, and contact emergency services."
}
```

## Project Structure

```text
natural-disaster-llm-demo/
  requirements.txt
  data/
    train.jsonl
  train.py
  infer.py
  README.md
```

## Setup

Create and activate a virtual environment.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements

`requirements.txt`

```txt
torch
transformers
datasets
trl
peft
accelerate
```

## Dataset

Training data is stored in:

```text
data/train.jsonl
```

Each row contains:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a disaster response assistant..."
    },
    {
      "role": "user",
      "content": "Heavy rain caused flooding..."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "{\"disaster_type\":\"flood\",\"severity\":\"high\",\"recommended_action\":\"Move to higher ground...\"}"
    }
  ]
}
```

## Train

Run:

```bash
python train.py
```

After training, the LoRA adapter will be saved to:

```text
natural-disaster-lora/
```

## Run Inference

Run:

```bash
python infer.py
```

Example input:

```text
The river overflowed after nonstop rain, and nearby houses are surrounded by water.
```

Example output:

```json
{
  "disaster_type": "flood",
  "severity": "high",
  "recommended_action": "Move to higher ground if safe, avoid floodwater, and contact emergency services for evacuation support."
}
```

## Notes

This is a small educational demo. It is not meant for real emergency decision-making.

For real disaster response, always follow official emergency services and local authorities.

# GOLD - Product Requirements Document
## Personal On-Policy Distillation Setup

---

## 1. Executive Summary

### Vision
A simple, self-hosted toolkit for personal on-policy distillation experiments. Compress large language models into smaller, faster versions for specific personal use cases while maintaining quality.

### Problem Statement
As an individual practitioner, you face:
- **High inference costs**: Running large models (70B+) for personal projects is expensive
- **API dependencies**: Relying on OpenAI/Anthropic APIs adds recurring costs and latency
- **Limited customization**: Can't easily adapt models for your specific domain or style
- **Complex tooling**: Existing distillation frameworks are enterprise-focused and overcomplicated

### Solution
A lightweight, personal distillation setup that:
- **Simple to run**: Docker-based setup, runs on single GPU or cloud instance
- **Cost-effective**: One-time distillation cost, then cheap inference forever
- **Flexible**: Works with any HuggingFace models, customize for your needs
- **Transparent**: Clear visibility into costs, quality, and progress

---

## 2. Understanding On-Policy Distillation

### The Core Concept

**Traditional Approach:**
- You: "I want a small model that writes like GPT-4"
- Solution: Fine-tune on GPT-4 outputs
- Problem: Your model faces different contexts than GPT-4 did, leading to quality drift

**On-Policy Distillation:**
- Your small model generates text
- GPT-4 (teacher) grades each token: "good" or "bad"
- Your model learns from its own mistakes
- Result: 80-95% of teacher quality at 10-30x lower cost

### Why It Matters for Personal Use

**Example Use Cases:**

1. **Personal Writing Assistant**
   - Teacher: Claude 3.5 Sonnet (via API)
   - Student: Llama 3.2 8B (local)
   - Goal: Match your writing style for emails, docs
   - Cost: $50 training → $0/month inference vs $20+/month API

2. **Code Explanation Bot**
   - Teacher: GPT-4 or Claude
   - Student: CodeLlama 7B
   - Goal: Explain your codebase in your preferred style
   - Cost: $30 training → Free local inference

3. **Domain-Specific Q&A**
   - Teacher: Mixtral 8x7B or GPT-4
   - Student: Mistral 7B
   - Goal: Answer questions about your specific domain (finance, biology, etc.)
   - Cost: $40 training → Free inference

---

## 3. Personal Use Case Specification

### Your Goals
- [ ] Train 1-3 distilled models per month
- [ ] Run on single GPU (or rent as needed)
- [ ] Keep training costs under $50-100/model
- [ ] Deploy models locally or on cheap inference endpoints
- [ ] Experiment with different teacher/student pairs

### Constraints
- **Budget**: $100-300/month total (training + inference)
- **Hardware**: Single consumer GPU (4090/A5000) or cloud rentals
- **Time**: Set up once, train overnight, use for months
- **Complexity**: Minimal DevOps, simple Python scripts preferred

---

## 4. Architecture (Simplified)

### Option A: Fully Local (Recommended for Privacy)

```
┌─────────────────────────────────────────────────┐
│         Your Local Machine                       │
│                                                  │
│  ┌──────────────────────────────────────┐      │
│  │  GOLD Training Script                │      │
│  │  - Loads student model (8B)          │      │
│  │  - Calls teacher API or local model  │      │
│  │  - Trains student on own outputs     │      │
│  └──────────────────────────────────────┘      │
│                                                  │
│  Storage:                                        │
│  - Training data: 10-50 GB                      │
│  - Checkpoints: 5-20 GB per model               │
│  - Final model: 5-15 GB                         │
└─────────────────────────────────────────────────┘

Hardware Needed:
- RTX 4090 (24GB): Can train 7-8B students ✓
- RTX 4080 (16GB): Can train 7B students with LoRA ✓
- Mac M2 Ultra: Can train 7B students (slower) ✓
```

### Option B: Hybrid (Cloud Training, Local Inference)

```
┌─────────────────────────────────────────────────┐
│         Cloud GPU (Vast.ai / Runpod)            │
│                                                  │
│  Train on rented A100/H100 ($1-2/hour)         │
│  - Faster training (4-8 hours)                  │
│  - Download trained model when done             │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         Your Local Machine                       │
│                                                  │
│  Run inference on trained model (Free!)         │
│  - RTX 3060 (12GB) or better                   │
│  - M1/M2 Mac (16GB+ RAM)                        │
│  - Even CPU-only (slower)                       │
└─────────────────────────────────────────────────┘
```

---

## 5. Features (Minimal Viable Setup)

### 5.1 Core Training Script

**What You Need:**
```python
# gold_trainer.py
from trl import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simple configuration
config = GOLDConfig(
    teacher_model="meta-llama/Llama-3.1-70B-Instruct",  # or API
    student_model="meta-llama/Llama-3.2-8B",
    dataset="your_custom_data",
    lambda_=0.5,  # 50% on-policy
    output_dir="./checkpoints"
)

trainer = GOLDTrainer(config)
trainer.train()  # Run overnight
```

**Features:**
- ✅ On-policy distillation with GKD
- ✅ Progress tracking (tqdm bars)
- ✅ Auto-checkpointing every N steps
- ✅ Evaluation on test set
- ✅ Cost tracking (API calls, GPU hours)
- ❌ No web UI (just terminal)
- ❌ No multi-user support
- ❌ No complex orchestration

### 5.2 Simple Monitoring (Optional)

**Jupyter Notebook Dashboard:**
```python
# monitor.ipynb
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
logs = pd.read_json("logs/training.jsonl", lines=True)

# Plot loss curves
plt.plot(logs['step'], logs['loss'])
plt.title('Training Loss')
plt.show()

# Show sample outputs
print("Student output:", logs.iloc[-1]['student_output'])
print("Teacher score:", logs.iloc[-1]['teacher_score'])
```

**Weights & Biases Integration (Free tier):**
- Automatic loss/metric logging
- Sample text generations
- GPU utilization tracking
- Compare multiple runs

### 5.3 Inference Script

```python
# inference.py
from transformers import pipeline

# Load your distilled model
model = pipeline("text-generation", model="./checkpoints/final")

# Use it!
output = model("Write me an email about...", max_length=200)
print(output[0]['generated_text'])
```

---

## 6. Model Recommendations & Sourcing

### Teacher Models (Choose One)

#### Option 1: API-Based Teachers (Easiest)
**Best for:** Quick experiments, highest quality

| Model | Provider | Cost/1M tokens | Quality | Use Case |
|-------|----------|----------------|---------|----------|
| **GPT-4o** | OpenAI | $2.50 | Excellent | General purpose, coding |
| **Claude 3.5 Sonnet** | Anthropic | $3.00 | Excellent | Writing, analysis |
| **GPT-4o mini** | OpenAI | $0.15 | Very Good | Budget option |
| **Gemini 1.5 Pro** | Google | $1.25 | Very Good | Long context |

**Pros:**
- No GPU needed for teacher
- Always latest/best models
- Simple API integration

**Cons:**
- Ongoing API costs during training ($10-50)
- Rate limits
- Data sent to external service

**Cost Estimate:**
- Training 10k samples, 512 tokens avg = 5M tokens
- GPT-4o: 5M × $2.50/1M = **$12.50**
- GPT-4o mini: 5M × $0.15/1M = **$0.75**

#### Option 2: Local Open-Source Teachers (Private)
**Best for:** Privacy, unlimited experimentation

| Model | Size | VRAM Needed | Quality | Where to Get |
|-------|------|-------------|---------|--------------|
| **Llama 3.1 70B** | 70B | 80GB (A100) | Excellent | HuggingFace |
| **Mixtral 8x7B** | 47B | 48GB | Very Good | HuggingFace |
| **Qwen 2.5 72B** | 72B | 80GB | Excellent | HuggingFace |
| **Llama 3.1 70B (quantized)** | 70B | 40GB (4-bit) | Very Good | HuggingFace |

**Pros:**
- No API costs (free!)
- Complete privacy
- Unlimited tokens

**Cons:**
- Need powerful GPU or rent cloud GPU
- Slower inference
- Managing large models

**Cost Estimate (Cloud):**
- Rent A100 80GB: $1.50/hour (Vast.ai)
- Training time: 8 hours
- Total: **$12/training run**

#### Option 3: Medium-Size Local Teachers (Balanced)
**Best for:** Single consumer GPU, good quality

| Model | Size | VRAM Needed | Quality | Notes |
|-------|------|-------------|---------|-------|
| **Qwen 2.5 32B** | 32B | 24GB (4090) | Very Good | Fits on 4090 |
| **Mixtral 8x22B (quantized)** | 141B | 24GB (4-bit) | Excellent | Slow but works |
| **Llama 3.1 8B** | 8B | 10GB | Good | Fast teacher |

**Pros:**
- Runs on consumer GPU
- Free inference
- Good enough for many tasks

**Cons:**
- Lower quality than 70B+
- Still need decent GPU

### Student Models (Your Distilled Model)

**Recommended Sizes:**

| Model | Size | VRAM (Training) | VRAM (Inference) | Speed | Use Case |
|-------|------|-----------------|------------------|-------|----------|
| **Llama 3.2 3B** | 3B | 12GB (LoRA) | 4GB | Very Fast | Mobile, edge |
| **Llama 3.2 8B** | 8B | 16GB (LoRA) | 8GB | Fast | General purpose |
| **Mistral 7B** | 7B | 16GB (LoRA) | 7GB | Fast | Chat, reasoning |
| **Qwen 2.5 7B** | 7B | 16GB (LoRA) | 7GB | Fast | Multilingual |
| **CodeLlama 7B** | 7B | 16GB (LoRA) | 7GB | Fast | Code-specific |

**Rule of Thumb:**
- 3B models: 70-80% of teacher quality
- 7-8B models: 80-90% of teacher quality
- 13B+ models: 85-95% of teacher quality

### Where to Source Models

1. **HuggingFace Hub** (Primary)
   - URL: https://huggingface.co/models
   - Filter: `Tasks: Text Generation`, `Sort: Most Downloads`
   - Look for: High download count, recent updates, good documentation

2. **Recommended Starting Point:**
   ```python
   # Teacher options (pick one):
   teacher = "meta-llama/Llama-3.1-70B-Instruct"  # if you have A100
   teacher = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # good balance
   teacher = "openai/gpt-4o"  # via API

   # Student (pick one):
   student = "meta-llama/Llama-3.2-8B"  # best general purpose
   student = "mistralai/Mistral-7B-v0.3"  # good alternative
   student = "Qwen/Qwen2.5-7B"  # if you need multilingual
   ```

3. **Model Cards to Check:**
   - License (commercial use allowed?)
   - Benchmark scores (MMLU, GSM8K)
   - Language support
   - Context length

---

## 7. Cost Estimates (Realistic)

### Scenario 1: Budget Setup (Under $100/month)

**Hardware:** Your own RTX 4090 (already owned)

**Costs:**
- Teacher: GPT-4o mini API: **$5/training run**
- Electricity: 450W × 8 hours × $0.15/kWh = **$0.54**
- Dataset prep: OpenAI embeddings (optional): **$2**
- **Total per model: $7.54**

**Monthly:** 3 models = **$22.62**
**Electricity (running inference): ~$10/month**
**Grand total: ~$32.62/month**

**Break-even vs API:**
- Claude API: $20/month subscription
- You break even in: **1 month**
- After that: Pure savings

### Scenario 2: Cloud Training (No GPU)

**Hardware:** Rent on-demand

**Costs per training run:**
- GPU rental: A100 40GB × 8 hours × $1.39/hour (Vast.ai): **$11.12**
- Teacher: GPT-4o API × 5M tokens: **$12.50**
- Storage: 50 GB × $0.10/GB/month: **$5**
- **Total: $28.62 per model**

**Monthly:** 2 models = **$57.24**
**Inference:** Local on laptop (free) or Modal ($0.01/min)
**Grand total: ~$60-70/month**

**Break-even:**
- vs Claude Pro ($20/month) + GPT-4 API ($50/month) = $70/month
- You break even in: **1 month**

### Scenario 3: Premium Setup (Best Quality)

**Hardware:** Rent A100 80GB for teacher + student

**Costs per training run:**
- GPU rental: A100 80GB × 12 hours × $1.89/hour: **$22.68**
- Teacher: Llama 3.1 70B (local, free!)
- Storage: 100 GB: **$10**
- **Total: $32.68 per model**

**Monthly:** 4 models = **$130.72**
**Inference:** Local or cloud at ~$10/month
**Grand total: ~$140/month**

**Quality:** 90-95% of GPT-4 level
**Privacy:** 100% (no data leaves your control)

### Scenario 4: Ultra-Budget (Learning/Experimentation)

**Hardware:** Google Colab Pro ($12/month) or Kaggle (free)

**Costs:**
- Colab Pro: **$12/month**
- Teacher: GPT-4o mini: **$2/run**
- **Total: $12 + ($2 × 2 models) = $16/month**

**Limitations:**
- 24-hour training limit
- Smaller models only (7B max)
- Less reliable

---

## 8. Recommended Setup Path

### Phase 1: Proof of Concept (Week 1)

**Goal:** Train your first distilled model, validate quality

**Steps:**
1. **Set up environment:**
   ```bash
   git clone https://github.com/huggingface/trl
   cd trl
   pip install -e .
   pip install openai anthropic
   ```

2. **Prepare small dataset (1000 examples):**
   - Your own prompts/completions
   - Or use existing: `ultrachat_200k`, `orca-math`, etc.

3. **Configure training:**
   ```python
   config = GOLDConfig(
       teacher_model="openai/gpt-4o-mini",  # Start cheap
       student_model="meta-llama/Llama-3.2-3B",  # Small
       lambda_=0.3,  # 30% on-policy (conservative)
       max_steps=500,  # Quick test
   )
   ```

4. **Train (2-4 hours):**
   ```bash
   python examples/scripts/gold_trainer.py --config config.yaml
   ```

5. **Evaluate:**
   - Compare outputs side-by-side
   - Run on your specific use case
   - Measure quality drop (aim for <20%)

**Expected Cost:** $5-10
**Expected Quality:** 70-80% of GPT-4o mini

### Phase 2: Production Model (Week 2-3)

**Goal:** Train high-quality model for daily use

**Improvements:**
1. Larger dataset (10k examples)
2. Better teacher (GPT-4o or Claude)
3. Bigger student (8B)
4. More on-policy (lambda=0.5)
5. Longer training (2000 steps)

**Expected Cost:** $30-50
**Expected Quality:** 85-92% of teacher

### Phase 3: Optimization (Ongoing)

**Goal:** Reduce costs, improve quality

**Experiments:**
- Try different lambda values (0.3, 0.5, 0.7)
- Test local teachers (Mixtral, Qwen)
- Quantization (run 8B models on 8GB GPU)
- Longer training (does it plateau?)
- Domain-specific datasets

---

## 9. Technical Setup Guide

### 9.1 Prerequisites

**Software:**
- Python 3.10+
- CUDA 12.1+ (if using NVIDIA GPU)
- Git

**Accounts:**
- HuggingFace account (free)
- OpenAI or Anthropic API key (optional)
- Weights & Biases account (optional, free tier)

### 9.2 Installation

```bash
# Clone this repo
git clone <your-repo-url>
cd GOLD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install torch transformers trl datasets wandb openai anthropic
pip install flash-attn --no-build-isolation  # optional, speeds up training
```

### 9.3 Project Structure

```
GOLD/
├── README.md                 # Quick start guide
├── GOLD_PRD.md              # This document
├── requirements.txt          # Python dependencies
├── config/
│   ├── default.yaml         # Default training config
│   ├── quick_test.yaml      # Fast test run
│   └── production.yaml      # Full training
├── scripts/
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   └── inference.py         # Run trained model
├── data/
│   ├── prepare_dataset.py   # Format your data
│   └── samples/             # Example prompts
├── notebooks/
│   └── monitor.ipynb        # Training visualization
└── outputs/
    ├── checkpoints/         # Saved models
    └── logs/                # Training logs
```

### 9.4 Configuration File

**config/default.yaml:**
```yaml
# Teacher model
teacher:
  model_name: "openai/gpt-4o-mini"  # or HF model
  temperature: 0.7
  max_tokens: 512

# Student model
student:
  model_name: "meta-llama/Llama-3.2-8B"
  use_lora: true  # More memory efficient
  lora_r: 16
  lora_alpha: 32

# Distillation settings
distillation:
  strategy: "on_policy"
  lambda: 0.5  # 50% on-policy samples
  beta: 0.5    # GKD interpolation

# Training
training:
  max_steps: 2000
  batch_size: 4
  gradient_accumulation: 8  # effective batch = 32
  learning_rate: 5e-5
  warmup_steps: 100
  save_steps: 250
  eval_steps: 250

# Dataset
dataset:
  name: "your-dataset"  # HF dataset or local path
  split: "train"
  max_samples: 10000

# Output
output:
  dir: "./outputs/run_1"
  wandb_project: "gold-distillation"  # optional
```

---

## 10. Evaluation & Quality Checks

### Quick Validation

**After training, test on your specific use case:**

```python
# Load models
teacher = load_model("gpt-4o-mini")
student = load_model("./outputs/final")

# Test prompts
prompts = [
    "Explain quantum computing in simple terms",
    "Write a professional email apologizing for a delay",
    "Debug this Python code: [code]"
]

# Compare outputs
for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    teacher_output = teacher(prompt)
    student_output = student(prompt)

    print(f"\nTeacher:\n{teacher_output}")
    print(f"\nStudent:\n{student_output}")

    # Manual scoring
    score = input("Rate student (1-5): ")
```

### Quantitative Metrics

**If you have labeled test set:**

```python
from sklearn.metrics import accuracy_score
from evaluate import load

# Perplexity (lower = better)
perplexity = load("perplexity")
ppl = perplexity.compute(predictions=student_outputs, references=references)

# ROUGE (for summarization)
rouge = load("rouge")
scores = rouge.compute(predictions=student_outputs, references=references)

# Human evaluation (best but time-consuming)
# Sample 50-100 outputs, rate on scale 1-5
```

**Target Metrics:**
- Perplexity: Within 10-20% of teacher
- ROUGE-L: >0.4 for summaries
- Human eval: >4/5 on average
- Task-specific: 80-90% of teacher performance

---

## 11. Cost Optimization Tips

### 1. Use Cheaper Teachers
- Start with GPT-4o mini ($0.15/M tokens) instead of GPT-4o ($2.50/M)
- Or use local Mixtral 8x7B (free but need GPU)

### 2. Reduce Dataset Size
- 1k samples: Good for narrow tasks
- 5k samples: Good for general quality
- 10k+ samples: Diminishing returns

### 3. Use LoRA Instead of Full Fine-tuning
```python
# LoRA uses 1/4 the memory
config.use_lora = True
config.lora_r = 16  # Lower = less memory, slightly lower quality
```

### 4. Rent Spot Instances
- Vast.ai: 50-70% cheaper than on-demand
- RunPod: Good spot availability
- Lambda Labs: Simple, but less cheap

### 5. Quantize Models
```python
# 4-bit quantization = 1/4 memory
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-8B",
    load_in_4bit=True,  # 8B fits in 8GB!
)
```

### 6. Reuse Teacher Outputs (Hybrid)
```python
# Generate 1000 teacher outputs once
teacher_outputs = generate_batch(teacher, prompts)
save_cache(teacher_outputs)

# Use 70% cached (off-policy) + 30% fresh (on-policy)
config.lambda_ = 0.3
config.cached_outputs = "teacher_cache.jsonl"
```

**Savings:** Reduce API costs by 70%

---

## 12. Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
```python
# 1. Reduce batch size
config.batch_size = 2  # or even 1

# 2. Increase gradient accumulation
config.gradient_accumulation = 16  # keeps effective batch size

# 3. Use LoRA
config.use_lora = True

# 4. Enable gradient checkpointing
config.gradient_checkpointing = True

# 5. Reduce sequence length
config.max_length = 256  # instead of 512
```

### Issue: Training is Too Slow

**Solutions:**
- Use Flash Attention: `pip install flash-attn`
- Reduce eval frequency: `config.eval_steps = 500`
- Use fewer on-policy samples: `config.lambda_ = 0.3`
- Rent faster GPU (A100 > A6000 > RTX 4090)

### Issue: Quality is Poor

**Check:**
1. Is student too small? Try 8B instead of 3B
2. Is lambda too high? Try 0.3-0.5 instead of 0.7+
3. Is dataset diverse enough? Add more examples
4. Is training long enough? Try 2-3 epochs
5. Is teacher good enough? Compare teacher vs ground truth

---

## 13. Next Steps After First Model

### Option 1: Deploy Locally

```python
# Serve with FastAPI
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model = pipeline("text-generation", model="./outputs/final")

@app.post("/generate")
def generate(prompt: str):
    return model(prompt, max_length=200)
```

### Option 2: Deploy to Cloud (Cheap)

**Modal.com ($0.01/minute when running):**
```python
import modal

stub = modal.Stub("my-distilled-model")

@stub.function(gpu="T4")
def generate(prompt):
    model = load_model("./outputs/final")
    return model(prompt)
```

**Cost:** $0.01/min × 60 min/month = **$0.60/month**

### Option 3: Quantize for Edge

```python
# Export to GGUF for llama.cpp
python convert.py ./outputs/final --outfile model.gguf --quant q4_k_m

# Run on MacBook/phone/Raspberry Pi
./llama.cpp -m model.gguf -p "Your prompt"
```

---

## 14. Learning Resources

### Documentation
- HuggingFace TRL: https://huggingface.co/docs/trl
- GOLD Trainer: https://huggingface.co/docs/trl/main/en/gold_trainer
- PyTorch Docs: https://pytorch.org/docs/

### Papers
- "On-Policy Distillation" (Agarwal et al., 2023): https://arxiv.org/abs/2306.13649
- "Generalized Knowledge Distillation" (HuggingFace): Check TRL docs

### Communities
- HuggingFace Discord: https://hf.co/join/discord
- r/LocalLLaMA (Reddit): Great for practical tips
- r/MachineLearning: More academic

### Example Repos
- TRL examples: https://github.com/huggingface/trl/tree/main/examples
- Distillation tutorials: Search "LLM distillation" on GitHub

---

## 15. FAQ

**Q: Can I distill GPT-4 into a 3B model?**
A: Yes, but expect 70-80% quality. For 85-90%, use 8B+ student.

**Q: Is my data sent to OpenAI if I use GPT-4o as teacher?**
A: Yes, prompts go through their API. Use local teacher for privacy.

**Q: How long does training take?**
A: 4-12 hours on single GPU, depending on dataset size and model.

**Q: Can I use this commercially?**
A: Check licenses: LLama 3.2 (yes), Mistral (yes), teacher models (depends).

**Q: What if I don't have a GPU?**
A: Rent one! Vast.ai is cheapest (~$0.30/hour for RTX 3090).

**Q: Can I distill into non-English models?**
A: Yes! Use Qwen, Aya, or other multilingual models.

**Q: How much worse is the student vs teacher?**
A: Typically 10-20% worse on benchmarks, but task-dependent. Test on YOUR use case.

---

## 16. Success Metrics (Personal)

### You'll know it's working if:
- ✅ Student outputs are "good enough" for your use case 80%+ of the time
- ✅ You've reduced API costs by 50-90%
- ✅ Inference latency is acceptable (<2 seconds)
- ✅ Model runs on your hardware without OOM
- ✅ You're actually using it daily (not just a toy)

### Red flags:
- ❌ Student constantly gives nonsense
- ❌ Training costs more than just using APIs
- ❌ Quality drops below 70% of teacher
- ❌ Too slow to be useful

---

## 17. Budget Summary Table

| Scenario | Setup Cost | Monthly Cost | Inference Cost | Quality | Best For |
|----------|-----------|--------------|----------------|---------|----------|
| **Ultra Budget** | $0 | $15 | Free (local) | 70-80% | Learning, experiments |
| **Own GPU** | $1500 (GPU) | $30 | Free (local) | 85-90% | Daily use, privacy |
| **Cloud Rental** | $0 | $60 | $0.01/min | 85-92% | No hardware, flexibility |
| **Premium** | $0 | $140 | Free (local) | 90-95% | Best quality, lots of models |

**Recommendation:** Start with **Cloud Rental** scenario ($60/month) to validate before buying GPU.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-01-04 | Rewrote for personal use case, added cost estimates and model recommendations |
| 1.0 | 2025-01-04 | Initial commercial PRD |

---

## Getting Started Checklist

- [ ] Read sections 2-3 (understand on-policy distillation)
- [ ] Choose your teacher model (section 6)
- [ ] Choose your student model (section 6)
- [ ] Estimate costs for your scenario (section 7)
- [ ] Set up environment (section 9.2)
- [ ] Prepare small test dataset (100 examples)
- [ ] Run quick test (section 8, Phase 1)
- [ ] Evaluate outputs (section 10)
- [ ] If good: Scale up to production dataset
- [ ] Deploy and use daily!

**Expected timeline:** 2-3 days for first model, then 1 day per additional model.

**Need help?** Open an issue in this repo or ask on HuggingFace Discord.

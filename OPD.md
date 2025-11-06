# On-Policy Distillation (OPD) Implementation Summary

## Overview

This document summarizes the complete discussion on implementing On-Policy Distillation (OPD) using HuggingFace's GOLD framework for training a personal investment analyst model.

**Goal:** Compress a large Qwen3 teacher model into a smaller Qwen3 student model while maintaining 85-92% quality for investment analysis tasks.

---

## What is On-Policy Distillation?

### Core Concept

On-policy distillation is a training technique where:
1. **Student model** (small, fast) generates outputs
2. **Teacher model** (large, smart) evaluates each token of student's output
3. Student learns from teacher's token-level feedback
4. This happens in student's own generated contexts (on-policy)

### Why It's Better Than Traditional Distillation

**Traditional (Off-Policy):**
- Student copies teacher-generated outputs
- Student never learns error correction
- Distribution mismatch at inference time
- Results in lower quality

**On-Policy (GOLD):**
- Student generates → teacher grades
- Student learns from its own mistakes
- Training distribution matches inference distribution
- Results in better quality (85-92% vs 70-80%)

### Chess Analogy

Like learning chess:
- **Traditional:** Watch grandmaster play, memorize moves
- **On-Policy:** Play yourself, grandmaster grades each move you make

---

## Your Use Case: Investment Analyst Model

### Dataset

- **File:** `on_policy_prompts.jsonl` (119 prompts)
- **Location:** Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/
- **Format:** JSONL with system + user messages
- **System Message:** Detailed investment analyst persona (1,231 chars)
- **User Prompts:** Investment analysis requests (166-209 chars each)
- **Topics:** Emerging markets, policy analysis, market implications, macro/micro analysis

### Quality Expectations

If you distill **Qwen3 32B → Qwen3 7B:**
- Student quality: **85-92% of Qwen3 32B**
- Cost: **$5.56 per training run**
- Training time: **4 hours**
- Inference: **Free locally**
- Break-even: **2 weeks** vs Claude API usage

---

## Recommended Setup: Qwen3 Models

### Why Qwen3?

- ✅ Excellent for reasoning (financial analysis)
- ✅ Multilingual support (English + Chinese)
- ✅ Available in multiple sizes (7B, 14B, 32B, 72B)
- ✅ Recent knowledge (better than Llama)
- ✅ Strong at data analysis tasks

### Recommended Configuration

**Teacher:** Qwen3 32B
**Student:** Qwen3 7B
**Framework:** HuggingFace TRL GOLDTrainer
**Training Location:** Cloud GPU (Lambda Labs A100 40GB)

```yaml
teacher:
  model_name: "Qwen/Qwen3-32B-Instruct"
  use_api: false  # Local model, not cloud API
  vram_needed: 24GB

student:
  model_name: "Qwen/Qwen3-7B-Instruct"
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  vram_needed: 8GB (with LoRA)

distillation:
  strategy: "on_policy"
  lambda: 0.4  # 40% on-policy, 60% off-policy
  beta: 0.5    # GKD interpolation
  use_uld_loss: false

training:
  max_steps: 1000
  batch_size: 2
  gradient_accumulation: 4
  learning_rate: 2e-5
  warmup_steps: 50
  eval_steps: 100
  save_steps: 100
  early_stopping_patience: 3

dataset:
  file: "on_policy_prompts.jsonl"
  max_samples: 119
  train_test_split: 91/10  # Keep 10 for testing

output:
  dir: "./outputs"
  wandb_project: "gold-investment-analysis"
```

---

## Complete Training Flow

### Step 1: Data Preparation
- Have your 119 prompts ready in JSONL format ✓
- Hold out 10 prompts for testing (train on 109)
- Each prompt: ~350 input tokens, ~400 output tokens

### Step 2: Training Loop (4-6 hours on A100 40GB)

**For each of 1000 training steps:**

1. **Load batch** (2 prompts) from your 119
2. **Student generates** output (~400 tokens)
3. **Teacher evaluates** each token's quality (log probabilities)
4. **Loss calculation** combines:
   - On-policy loss (student's own outputs, 40%)
   - Off-policy loss (cached teacher outputs, 60%)
5. **Backpropagation** updates student weights
6. **Loss decreases** from ~4.0 → ~0.5 over 1000 steps

**Loss curve progression:**
```
Step 0:     Loss = 4.50 (student untrained)
Step 100:   Loss = 2.45 (learning!)
Step 250:   Loss = 1.89 (improving)
Step 500:   Loss = 1.12 (halfway)
Step 750:   Loss = 0.78 (converging)
Step 1000:  Loss = 0.52 (done!)
```

### Step 3: Validation & Early Stopping
- Validation loss monitored every 100 steps
- If validation doesn't improve for 3 checks → stop training
- Best model saved automatically
- Prevents overfitting

### Step 4: Download Model
- Download trained Qwen3 7B from cloud GPU
- File size: ~8GB (full PyTorch weights)

### Step 5: Convert to GGUF (for LM Studio)
- Convert PyTorch → GGUF format
- Quantize to 4-bit (reduces to ~4GB)
- Makes inference faster on Mac

### Step 6: Load in LM Studio
- Open LM Studio (free app)
- Load your GGUF model
- Chat with your distilled investment analyst model

### Step 7: Evaluate Quality
- Test on held-out 10 prompts
- Compare student vs Qwen3 32B outputs
- Rate quality 1-5
- Target: >4/5 on your specific use case

### Step 8: Feedback Loop
If quality is good:
- ✅ Deploy and use daily
- ✅ Model is done

If quality needs improvement:
- Try different lambda (0.3, 0.5, 0.7)
- Use better teacher (Qwen3 72B)
- Add more training data
- Re-train

---

## Cost Breakdown

### Hardware Options

**Option A: Local Mac (Slow)**
```
GPU: Your M-series Mac (free)
Training time: 12-24 hours
Cost: $0
Speed: Very slow, frustrating
```

**Option B: Cloud GPU (Recommended)**
```
GPU: Lambda Labs A100 40GB ($1.39/hr)
Training time: 4 hours
Cost: $5.56 (GPU)
Speed: Fast, practical
```

### Total Costs

| Component | Cost |
|-----------|------|
| A100 40GB × 4 hours | $5.56 |
| Qwen3 models (free from HF) | $0 |
| Anthropic/OpenAI APIs | $0 (using local Qwen3) |
| Storage (model files) | Negligible |
| **Total per training run** | **$5.56** |

**Break-even analysis:**
- Claude API: $0.10-0.15 per analysis
- 119 prompts × $0.10 = $11.90 per full pass
- 1 training run ($5.56) ≈ cost of 55 analyses via API
- Break-even: ~2 weeks of normal use
- After that: Pure savings

---

## Why NOT MLX

Your existing MLX fine-tuning GUI is excellent for traditional LoRA fine-tuning, but **can't do OPD** because:

1. **MLX is Apple-specific framework**
   - Optimized for M-series Metal GPU only
   - Limited to small model inference/fine-tuning
   - No GOLD trainer support

2. **OPD requires PyTorch + HuggingFace**
   - Need complex training loop (teacher + student)
   - Need token-level feedback mechanism
   - MLX framework too lightweight

3. **Solution:** Use separate PyTorch backend
   - Keep your MLX GUI for regular SFT fine-tuning
   - Add new OPD tab using HF GOLDTrainer
   - Two complementary training pipelines

---

## HuggingFace GOLD Framework

### What is GOLD?

**GOLD = Generative On-policy Learning and Distillation**

- Part of HuggingFace TRL (Transformers Reinforcement Learning) library
- Built-in on-policy distillation trainer
- Subclasses SFTTrainer
- Handles all the complex training logic for you

### What It Does

```python
from trl import GOLDConfig, GOLDTrainer

config = GOLDConfig(
    teacher_model="Qwen/Qwen3-32B-Instruct",
    student_model="Qwen/Qwen3-7B-Instruct",
    lambda_=0.4,  # 40% on-policy
    # ... other configs
)

trainer = GOLDTrainer(config)
trainer.train()  # Everything happens here
```

### What It Handles Automatically

- ✅ Loading teacher and student models
- ✅ Student generation loop
- ✅ Teacher evaluation of student outputs
- ✅ Loss calculation (KL divergence + on-policy/off-policy blend)
- ✅ Backpropagation and weight updates
- ✅ Checkpointing and resuming
- ✅ Evaluation and early stopping
- ✅ Logging and monitoring

---

## Implementation Plan

### Phase 1: Setup (Day 1)
- [ ] Copy your 119 prompts to new OPD repo
- [ ] Create GOLDConfig with Qwen3 32B + 7B
- [ ] Create training script using GOLDTrainer
- [ ] Create Lambda Labs account
- [ ] Set up SSH keys

### Phase 2: Validation Run (Day 1-2)
- [ ] Rent A100 40GB on Lambda Labs ($1.39/hr)
- [ ] Upload training files to instance
- [ ] Run full training (4 hours)
- [ ] Download trained model

### Phase 3: Evaluation (Day 2)
- [ ] Convert model to GGUF format
- [ ] Load in LM Studio
- [ ] Test on held-out 10 prompts
- [ ] Compare with Qwen3 32B outputs
- [ ] Rate quality

### Phase 4: Deployment (Day 3)
- [ ] If good quality: Use daily
- [ ] If needs improvement: Adjust config and re-train
- [ ] Build inference interface (FastAPI/simple Python)

---

## Key Hyperparameters Explained

| Parameter | Value | What It Does |
|-----------|-------|-------------|
| **lambda_** | 0.4 | 40% on-policy (student), 60% off-policy (teacher) |
| **beta** | 0.5 | GKD interpolation (balance loss terms) |
| **learning_rate** | 2e-5 | How fast model learns (small = careful steps) |
| **batch_size** | 2 | How many prompts per gradient step |
| **max_steps** | 1000 | Total training iterations (~8 epochs through 119 prompts) |
| **eval_steps** | 100 | Check validation loss every 100 steps |
| **early_stopping_patience** | 3 | Stop if no improvement for 3 evaluations |

### How to Know If Hyperparameters Are Good

**Good signs:**
- Loss smoothly decreases from 4.0 → 0.5
- No sudden spikes or NaN values
- Validation loss follows training loss
- Training completes in 4-6 hours

**Bad signs:**
- Loss stuck (barely changes) → learning rate too low
- Loss becomes NaN → learning rate too high
- Loss jumpy/noisy → batch size too small
- Loss increases after improvement → lambda too high (overfitting)

---

## Model Comparison Options

### Student Model Choices

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| Qwen3 7B | 7B | Very Fast | 85-92% | Production use |
| Qwen3 14B | 14B | Fast | 88-95% | Higher quality |
| Llama 3.2 8B | 8B | Very Fast | 80-88% | General purpose |
| Mistral 7B | 7B | Very Fast | 82-90% | Alternative |

### Teacher Model Choices

| Model | Size | Quality | Cost (local) | Cost (API) |
|-------|------|---------|--------------|-----------|
| Qwen3 32B | 32B | Excellent | Free (24GB) | N/A |
| Qwen3 72B | 72B | Excellent | Free (80GB) | N/A |
| Claude 3.5 Sonnet | - | Excellent | N/A | $0.84 |
| GPT-4o | - | Excellent | N/A | $0.58 |
| GPT-4o mini | - | Very Good | N/A | $0.03 |

**Recommendation:** Qwen3 32B teacher (free, excellent quality, no API costs)

---

## Deployment Options After Training

### Option 1: LM Studio (Recommended)
- Load GGUF model in beautiful UI
- Chat interface
- Zero coding required
- Runs on MacBook

### Option 2: FastAPI Server
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model = pipeline("text-generation", model="./outputs/final")

@app.post("/generate")
def generate(prompt: str):
    return model(prompt, max_length=600)
```

### Option 3: Modal.com (Cloud Inference)
- Deploy model to cloud
- Cost: $0.01/minute when running
- Scale automatically
- No server management

---

## Next Steps

When starting the OPD repo with this summary:

1. **Create new repo** called `OPD`
2. **Copy this summary** as `OPD.md`
3. **Ask Claude to create:**
   - Training script (`train.py`)
   - Config file (`config.yaml`)
   - Lambda Labs setup guide
   - Evaluation script
   - GGUF conversion script
4. **You execute:**
   - Rent GPU on Lambda Labs
   - Run training
   - Download and evaluate model

---

## Resources

- **HuggingFace TRL GOLD:** https://huggingface.co/docs/trl/main/en/gold_trainer
- **On-Policy Distillation Paper:** https://arxiv.org/abs/2306.13649
- **Lambda Labs GPU Rental:** https://lambdalabs.com
- **LM Studio:** https://lmstudio.ai/
- **HuggingFace Models:** https://huggingface.co/models

---

## Questions for Claude in OPD Repo

When you open the new repo with this summary, ask Claude:

1. "Create the complete training script using HF GOLDTrainer"
2. "Create the config file for Qwen3 32B teacher + 7B student"
3. "How do I set up Lambda Labs and upload training files?"
4. "Create evaluation script to test quality on my held-out prompts"
5. "How do I convert the trained model to GGUF for LM Studio?"
6. "Create a simple FastAPI inference server"

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Goal** | Distill Qwen3 32B → 7B for investment analysis |
| **Dataset** | 119 investment analysis prompts |
| **Framework** | HuggingFace TRL GOLDTrainer |
| **Teacher** | Qwen3 32B (local) |
| **Student** | Qwen3 7B (trained) |
| **Training Hardware** | Cloud A100 40GB |
| **Training Time** | 4-6 hours |
| **Training Cost** | $5.56 |
| **Expected Quality** | 85-92% of Qwen3 32B |
| **Inference Cost** | Free (local) |
| **Break-even Time** | ~2 weeks vs API |
| **Deployment** | LM Studio on MacBook |

---

**Document Created:** November 4, 2025
**Status:** Ready for implementation in new OPD repo

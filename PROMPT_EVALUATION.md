# On-Policy Prompts Evaluation Report

## Overview

**File:** `on_policy_prompts.jsonl`
**Location:** Arjun LLM Fine Tuner
**Date:** November 4, 2025

---

## Dataset Statistics

### Size
- **Total prompts:** 119
- **Format:** JSONL (JSON Lines) - one prompt per line
- **Message structure:** Each prompt has exactly 2 messages (system + user)

### Content Analysis

**System Messages:**
- Count: 119 (one per prompt)
- Length: ~1,231 characters (consistent across all prompts)
- Role: System persona definition
- **Content:** Investment analyst persona with detailed writing style guidelines

**User Messages:**
- Count: 119
- Average length: 179 characters (range: 166-209)
- Role: Investment analysis requests
- **Topics:** Investment research, market analysis, policy analysis

---

## Use Case Identification

### Domain
**Investment Analysis & Research**

Your prompts are specialized for training a model that writes **sophisticated investment analyses**. This is a very specific, high-value use case.

### System Prompt (Investment Analyst Persona)

The system message defines an investment analyst with these characteristics:

**TONE:**
- Analytical and measured
- Balances confidence with epistemic humility
- Direct but not casual, authoritative but not arrogant

**STRUCTURE:**
- Investment thesis → systematic analysis → risk acknowledgment → conviction-weighted conclusion
- Data-driven with specific quantitative evidence

**VOCABULARY:**
- Academic but accessible
- Extensive domain terminology
- Precise technical terms

**SENTENCE PATTERNS:**
- Complex analytical sentences (25-35 words) for frameworks
- Short declarative statements (12-18 words) for emphasis
- Frequent dependent clauses for nuance

**ANALYTICAL APPROACH:**
- Risk-reward analysis frames
- Multiple perspectives before concluding
- Explicit about assumptions and limitations

### User Prompts

Your 119 prompts request analyses on diverse investment topics:

**Detected Categories:**
1. **Market analysis** - "What's your perspective on the investment insights described in..."
2. **Research synthesis** - "Write an investment analysis on the topic discussed in..."
3. **Implication analysis** - "Analyze the market implications of the concept presented here..."
4. **Policy analysis** - References to "Policy Paralysis in India", "Asia Crisis"

**Topics observed:**
- Emerging markets (EM) analysis
- Policy impacts on markets
- Asset class comparisons
- Market cap and liquidity analysis
- ESG considerations
- Macro vulnerability assessment

---

## Suitability for On-Policy Distillation

### ✅ Strengths

1. **Clear persona definition**
   - System prompt is detailed and specific
   - Will help model learn your preferred writing style
   - Consistent across all 119 prompts (good for distillation)

2. **Specialized domain**
   - Investment analysis is a valuable, niche skill
   - Distilled model will be highly specialized
   - Good ROI: trained model solves a real problem

3. **Consistent format**
   - All prompts follow same structure
   - All responses would have similar expected tone/style
   - Good for on-policy training stability

4. **Moderate complexity**
   - Prompts require domain knowledge but not excessive length
   - Average 179 characters is reasonable (not too simple, not token-heavy)
   - Good token efficiency for training

### ⚠️ Considerations

1. **Limited volume (119 prompts)**
   - Small dataset for training
   - May result in overfitting if teacher outputs aren't diverse
   - Recommend: Augment with 50-100 synthetic prompts

2. **No ground truth responses included**
   - File contains prompts only, not expected completions
   - Will need teacher model to generate responses during training
   - Cost will depend on teacher model quality

3. **Narrow domain focus**
   - Highly specialized for investment analysis
   - Less general-purpose than broader datasets
   - Distilled model won't be useful for other domains

---

## Cost Estimate for Training

### Scenario A: Using GPT-4o mini (Budget)

**Setup:**
- Teacher: GPT-4o mini API ($0.15 per 1M input tokens)
- Student: Llama 3.2 8B
- Hardware: Cloud GPU rental (A100 40GB, Vast.ai)

**Token calculation:**
- System prompt length: ~1,231 chars ≈ 300 tokens
- User prompt length: ~179 chars ≈ 45 tokens
- Per prompt: ~345 input tokens

**For 119 prompts:**
- Total input tokens: 119 × 345 = **41,055 tokens**
- Teacher cost: 41,055 / 1,000,000 × $0.15 = **$0.006** (negligible)

**Expected output tokens** (teacher responses):
- Investment analysis responses: ~300-500 tokens each
- Conservative estimate: 119 × 400 = **47,600 output tokens**
- Output cost: 47,600 / 1,000,000 × $0.50 = **$0.024**

**Training costs:**
- GPU rental: A100 40GB × 4 hours × $1.39/hr = **$5.56**
- Storage: 1 GB = **$0.10**
- Teacher API calls: **$0.03** (negligible)

**Total: ~$5.69 per training run**

---

### Scenario B: Using Local Teacher (Privacy)

**Setup:**
- Teacher: Mixtral 8x7B (local, free!)
- Student: Llama 3.2 8B
- Hardware: A100 80GB rental (for both teacher + student)

**Costs:**
- GPU rental: A100 80GB × 8 hours × $1.89/hr = **$15.12**
- Storage: 2 GB = **$0.20**

**Total: ~$15.32 per training run**

**Advantage:** 100% data privacy, no API costs

---

### Scenario C: Using GPT-4o (High Quality)

**Setup:**
- Teacher: GPT-4o API ($2.50 per 1M input tokens, $10 per 1M output)
- Student: Llama 3.2 8B
- Hardware: A100 40GB (4 hours)

**Costs:**
- Input tokens: 41,055 / 1,000,000 × $2.50 = **$0.10**
- Output tokens: 47,600 / 1,000,000 × $10 = **$0.48**
- GPU rental: **$5.56**
- Storage: **$0.10**

**Total: ~$6.24 per training run**

**Better quality** (GPT-4o is superior to mini), minimal cost increase

---

## Recommended Training Setup

Given your 119 investment analysis prompts, here's what I recommend:

### For First Training Run (Validation)

**Teacher:** GPT-4o mini
**Student:** Llama 3.2 8B
**Hardware:** A100 40GB on Vast.ai
**Dataset:** All 119 prompts
**Lambda:** 0.4 (40% on-policy, 60% off-policy) - conservative for small dataset

**Expected:**
- Training time: 3-4 hours
- Cost: ~$6
- Quality: 80-85% of GPT-4o mini
- Output: Distilled investment analysis model

**Why this setup:**
- Cheap to validate quality
- Fast iteration
- Can assess if distilled model matches your writing style
- Low risk before investing in better teacher

### For Production Model (After Validation)

**Teacher:** GPT-4o (better quality analysis)
**Student:** Llama 3.2 8B or Mistral 7B
**Hardware:** A100 40GB or H100 (faster)
**Dataset:** 119 originals + 50-100 synthetic augmented prompts
**Lambda:** 0.5 (balanced on/off-policy)

**Expected:**
- Training time: 4-6 hours
- Cost: ~$8-10
- Quality: 85-92% of GPT-4o
- Output: High-quality investment analysis model

**Why upgrade:**
- GPT-4o writes better analyses than mini
- Augmented dataset reduces overfitting
- Higher lambda = better on-policy learning
- More balanced = more stable training

---

## Data Quality Assessment

### ✅ What's Good

1. **Consistent formatting** - All 119 prompts perfectly formatted JSON
2. **Proper schema** - system + user message structure matches OpenAI format
3. **Clear intent** - All prompts clearly ask for investment analysis
4. **Detailed persona** - System prompt is comprehensive (not generic)
5. **Diverse topics** - Emerging markets, policy, macro, micro analysis

### ⚠️ What Could Be Improved

1. **Sample prompts:**
   ```
   "Write an investment analysis on the topic discussed in this excerpt: Emerging Thoughts: Policy Paralysis in India..."
   ```
   - Text is truncated with "..."
   - Unclear if this is intentional or file corruption
   - **Action:** Verify the full user prompt content

2. **Missing diversity:**
   - All prompts have same system message
   - All are phrased as requests for analysis
   - Consider: Different personas? Shorter vs longer analyses?

3. **No evaluation set:**
   - All 119 prompts are training data
   - Recommend: Hold out 10-20 prompts for testing
   - Better assessment of model quality

---

## Implementation Roadmap

### Phase 1: Preparation (Day 1)
- [ ] Verify all 119 prompts are complete (no truncation)
- [ ] Hold out 10 prompts as test set (keep 109 for training)
- [ ] Create small validation set (5 prompts)
- [ ] Prepare config files

### Phase 2: First Training Run (Day 1-2)
- [ ] Train distilled model on 109 prompts
- [ ] Use GPT-4o mini as teacher (cheap validation)
- [ ] Cost: ~$6
- [ ] Time: 3-4 hours

### Phase 3: Evaluation (Day 2)
- [ ] Test on 10 held-out prompts
- [ ] Compare outputs with GPT-4o mini
- [ ] Measure quality (manual scoring 1-5)
- [ ] Assess if style matches your preferences

### Phase 4: Production Training (Day 2-3)
- [ ] If validation looks good: Train on full 119 prompts
- [ ] Use GPT-4o as teacher (better quality)
- [ ] Add 50-100 synthetic prompts to reduce overfitting
- [ ] Cost: ~$10-15
- [ ] Time: 4-6 hours

### Phase 5: Deployment (Day 3)
- [ ] Quantize model to GGUF format
- [ ] Run locally on MacBook or cloud
- [ ] Integration with your workflow

---

## Training Configuration Recommendation

**config.yaml for your dataset:**

```yaml
# Teacher model
teacher:
  model_name: "openai/gpt-4o-mini"  # Start cheap
  temperature: 0.8  # Investment analysis needs some creativity
  max_tokens: 600   # Investment analyses are typically 300-500 tokens

# Student model
student:
  model_name: "meta-llama/Llama-3.2-8B"
  use_lora: true    # Memory efficient
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

# Distillation settings
distillation:
  strategy: "on_policy"
  lambda: 0.4       # 40% on-policy (conservative for 119 samples)
  beta: 0.5         # GKD interpolation
  use_uld_loss: false

# Training
training:
  max_steps: 1000       # ~8 passes through 119 prompts
  batch_size: 2
  gradient_accumulation: 4   # Effective batch = 8
  learning_rate: 2e-5
  warmup_steps: 50
  save_steps: 100
  eval_steps: 100

# Dataset
dataset:
  name: "on_policy_prompts"
  split: "train"
  max_samples: 109      # Hold out 10 for testing
  train_test_split: 0.91

# Output
output:
  dir: "./outputs/investment_analyst_v1"
  wandb_project: "gold-investment-analysis"
```

---

## Recommendations Summary

### ✅ Do This

1. **Train with GPT-4o mini first** ($6)
   - Quick validation of concept
   - Assess quality match with your style

2. **Evaluate on test set** (10 prompts)
   - Manual scoring (rate 1-5 on analysis quality)
   - Check writing style matches your preferences
   - Verify no quality cliffs

3. **Augment dataset if needed**
   - If 119 prompts feel small: Add 50-100 synthetic variations
   - Different phrasings of similar analysis requests
   - Reduces overfitting

4. **Use GPT-4o for production** ($8-10)
   - Better analysis quality than mini
   - Minimal cost difference
   - Worthwhile upgrade

5. **Deploy locally**
   - Quantize to GGUF format
   - Run on MacBook with llama.cpp
   - Zero inference cost

### ❌ Don't Do This

1. **Don't use Llama 3.1 70B as teacher**
   - Overkill for 119 prompts
   - Would cost $50+ in GPU rental
   - GPT-4o mini is cheaper and sufficient

2. **Don't train on all 119 without test set**
   - Can't evaluate quality objectively
   - Recommendation: Hold out 10-20 prompts

3. **Don't go straight to production without validation**
   - Unknown if distilled model will match your writing style
   - Cheap validation first ($6), then scale up

---

## Next Steps

1. **Verify the prompt file**
   - Check if text truncations are intentional
   - Confirm all 119 prompts are complete

2. **Set up environment** (from GOLD_PRD.md)
   ```bash
   pip install torch transformers trl datasets openai
   ```

3. **Prepare API key**
   - OpenAI API key for teacher
   - Add to environment: `export OPENAI_API_KEY=...`

4. **Run first training** (this week)
   - Configure with provided config.yaml
   - Cost: $6, Time: 3-4 hours
   - Validate quality

5. **Iterate and improve** (ongoing)
   - Test different lambda values (0.3, 0.5, 0.7)
   - Try different student sizes (3B, 8B, 13B)
   - Measure ROI vs API usage

---

## ROI Analysis

**Current investment API costs:**
- Assume: 10 analyses per week at $0.10 each = **$1/week** = **$4.33/month**

**With distilled model:**
- Training: $6 one-time
- Inference: Free (local) or $0.01/month (cloud)
- Break-even: 1.5 months

**After 6 months:**
- API cost saved: $26
- Training cost: -$6
- Net savings: **+$20**

**After 1 year:**
- API cost saved: $52
- Training cost: -$6
- Net savings: **+$46**

**Plus benefits:**
- Zero latency (local inference)
- Zero API dependency
- Full privacy
- Customized to your style

---

## Questions to Consider

1. **Are the prompt texts complete?**
   - Some appear truncated in the sample
   - Verify before training

2. **Do you want only investment analysis?**
   - Or would general-purpose model be useful?
   - This dataset is highly specialized

3. **What quality is "good enough"?**
   - Define success criteria before training
   - Helps evaluate validation run

4. **How often will you use it?**
   - Daily use → qualitative break-even in weeks
   - Weekly use → qualitative break-even in months

---

## Files Generated

This analysis has been saved to: `/Users/macbook2024/FinRL/PROMPT_EVALUATION.md`

Ready to proceed with training? Let me know:
1. Verify the prompts are complete
2. Confirm you want to proceed with Phase 1 validation
3. Set up OpenAI API key
4. Choose hardware (cloud rental vs local GPU)

---

**Questions?** Ask me about:
- Model selection for your use case
- Cost optimization strategies
- Hardware recommendations
- Training configuration tuning

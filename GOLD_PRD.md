# GOLD - Product Requirements Document
## Generative On-policy Learning and Distillation Platform

---

## 1. Executive Summary

### Vision
Build a production-ready, user-friendly platform for on-policy knowledge distillation that enables practitioners to efficiently compress large language models into smaller, cost-effective versions while maintaining performance quality.

### Problem Statement
Current LLM distillation approaches suffer from:
- **Distribution mismatch**: Students learn from teacher contexts, not their own inference patterns
- **Complexity**: Existing tools lack intuitive interfaces for non-experts
- **Limited flexibility**: Hard to experiment with different distillation strategies
- **Poor monitoring**: Insufficient visibility into training dynamics and quality metrics

### Solution
A comprehensive platform that combines:
- **Advanced OPD Engine**: State-of-the-art on-policy distillation with GKD (Generalized Knowledge Distillation)
- **Clean Web UI**: Intuitive interface for configuring, monitoring, and managing distillation jobs
- **Cross-tokenizer support**: Enable distillation across different model families (e.g., LLaMA → Qwen)
- **Real-time analytics**: Track training metrics, cost savings, and quality benchmarks

---

## 2. Understanding On-Policy Distillation

### The Core Concept

On-policy distillation (OPD) is a powerful training technique that combines the best of two worlds:
- **On-policy relevance** from reinforcement learning (student learns in contexts it will actually encounter)
- **Dense reward signals** from traditional distillation (token-level feedback, not just endpoint scores)

### The Chess Analogy

Think of learning chess from a grandmaster. Traditional distillation is like watching the grandmaster play and trying to memorize their moves. On-policy distillation is like playing chess yourself while the grandmaster grades each of YOUR moves on a scale from "blunder" to "brilliant."

**Traditional Off-Policy Distillation:**
- Student watches teacher play chess
- Student tries to copy teacher's style
- Problem: Student faces positions the teacher never encountered
- Like learning to swim by watching videos

**On-Policy Distillation:**
- Student plays its own games
- Teacher evaluates each student move
- Student learns in contexts it will actually face
- Like learning to swim with a coach in the water

### Why It Matters

**Distribution Mismatch Problem**

When a student model is trained on teacher-generated data (off-policy), it learns to predict what the teacher would do in the teacher's contexts. But during inference, the student encounters its own generated contexts, which can differ significantly from the teacher's. This mismatch leads to:
- Compounding errors in multi-turn conversations
- Lower quality on edge cases
- Drift from intended behavior over long sequences

**OPD Solution**

By generating outputs from the student model itself and using the teacher to provide token-level feedback, OPD ensures:
- Student trains on distributions it will see at inference time
- Teacher provides dense, informative gradients for every token
- Best of RL (on-policy) + best of distillation (dense rewards)
- 9-30x cheaper than traditional RLHF in compute costs

### Key Applications

1. **Math Reasoning**: Student learns to check its own work, not just copy teacher solutions
2. **Assistant Models**: Multi-turn conversations where student's previous responses matter
3. **Code Generation**: Iterative refinement where student builds on its own code
4. **Long-form Writing**: Maintaining coherence in student's own writing style

---

## 3. Visual Overview

### On-Policy Distillation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│                                                              │
│  ┌──────────┐     Generate      ┌──────────┐              │
│  │ Student  │ ─────Outputs────> │ Student  │              │
│  │  Model   │                    │ Outputs  │              │
│  └──────────┘                    └──────────┘              │
│       ▲                               │                      │
│       │                               │                      │
│       │                               ▼                      │
│       │                          ┌──────────┐              │
│       │                          │ Teacher  │              │
│       │                          │  Model   │              │
│       │                          └──────────┘              │
│       │                               │                      │
│       │                               │                      │
│       │         Token-Level           │                      │
│       └────────Gradients──────────────┘                      │
│              (Dense Feedback)                                │
│                                                              │
│  Key Insight: Student learns from its OWN outputs,          │
│  not teacher's outputs. Teacher acts as a grading oracle.   │
└─────────────────────────────────────────────────────────────┘
```

### Cost & Quality Comparison

```
                    Traditional        On-Policy
                    Distillation      Distillation

Training Cost:      $$$$               $$
Inference Cost:     $$                 $$
Quality:            85%                92%
Multi-turn:         Fair               Excellent
Compute (FLOPs):    Baseline           10-30x reduction
```

---

## 4. Target Users

### Primary Personas

**1. ML Engineers**
- Need: Compress production models to reduce inference costs
- Pain: Complex tooling, lack of visibility into distillation quality
- Goal: 60-90% cost reduction while maintaining 85%+ performance

**2. AI Researchers**
- Need: Experiment with distillation strategies and compare techniques
- Pain: Hard to reproduce results, limited configuration options
- Goal: Publish reproducible distillation benchmarks

**3. AI Product Managers**
- Need: Monitor distillation jobs, track ROI, and approve model deployments
- Pain: No business-friendly dashboards
- Goal: Data-driven decisions on model compression strategies

---

## 5. Core Features

### 5.1 On-Policy Distillation Engine

#### Distillation Modes
- **On-Policy (Primary)**: Student generates outputs, teacher provides token-level feedback
- **Hybrid**: Mix of on-policy and off-policy data (configurable ratio via `lambda`)
- **Off-Policy**: Traditional distillation from fixed teacher outputs

#### Key Capabilities
- **Cross-Tokenizer Distillation**: Universal Logit Distillation (ULD) for different vocab sizes
- **Multi-Strategy Support**:
  - Forward KL divergence
  - Reverse KL divergence
  - Generalized JSD (Jensen-Shannon Divergence)
  - Token-level vs sequence-level distillation
- **Teacher Integration**:
  - Local HuggingFace models
  - vLLM-accelerated inference
  - API-based teachers (OpenAI, Anthropic, etc.)
  - Custom model endpoints

#### Configuration Parameters
```yaml
teacher:
  model_name: "meta-llama/Llama-3.1-70B-Instruct"
  use_vllm: true
  temperature: 0.7

student:
  model_name: "meta-llama/Llama-3.2-8B"
  learning_rate: 5e-5
  lora_config: {...}

distillation:
  strategy: "on_policy"  # on_policy, off_policy, hybrid
  lambda: 0.5  # on-policy sampling ratio
  beta: 0.5    # GKD interpolation
  use_uld_loss: true  # for cross-tokenizer
  seq_kd: false  # sequence vs token-level
```

### 5.2 Web Frontend (React + TypeScript)

#### Dashboard Views

**A. Job Creation Wizard**
```
Step 1: Select Models
├── Teacher Model (dropdown with search)
│   ├── HuggingFace Hub
│   ├── Local Models
│   └── API Endpoints
└── Student Model (same options)

Step 2: Configure Distillation
├── Strategy Selection (visual cards)
├── Hyperparameters (smart defaults with explanations)
└── Dataset Selection
    ├── Upload custom dataset
    ├── HuggingFace datasets
    └── Synthetic data generation

Step 3: Training Config
├── Hardware Selection (GPU count, type)
├── Cost Estimation (real-time)
├── Monitoring Webhooks (Slack, Discord, Email)
└── Auto-eval benchmarks (MMLU, GSM8K, etc.)

Step 4: Review & Launch
├── Configuration summary
├── Estimated runtime & cost
└── Launch button
```

**B. Training Dashboard**
```
Real-time Metrics Panel:
├── Loss curves (teacher loss, student loss, distillation loss)
├── Token-level accuracy
├── Generation samples (side-by-side teacher/student)
├── Resource utilization (GPU, memory, throughput)
└── Cost tracker ($ per hour, total spend)

Controls:
├── Pause/Resume
├── Early stopping
├── Checkpoint save
└── Hyperparameter adjustment (learning rate, temperature)
```

**C. Model Comparison**
```
Evaluation Tab:
├── Automatic benchmarks (MMLU, GSM8K, HumanEval)
├── Custom eval sets
├── A/B testing interface
└── Quality vs Cost tradeoff charts

Export:
├── Download model weights
├── Push to HuggingFace Hub
├── Deploy to inference endpoint
└── Generate model card
```

**D. Job History & Analytics**
```
Overview:
├── All distillation jobs (filterable table)
├── Success/failure rates
├── Total cost savings
└── Performance benchmarks

Per-Job Details:
├── Full configuration
├── Training curves
├── Final metrics
├── Artifacts (checkpoints, logs, model card)
└── Reproducibility info (git commit, environment)
```

### 5.3 Backend API (FastAPI)

#### Core Endpoints
```python
POST /api/v1/jobs/create
  - Create new distillation job
  - Returns: job_id, estimated_cost, estimated_duration

GET /api/v1/jobs/{job_id}
  - Get job status and metrics

POST /api/v1/jobs/{job_id}/pause
POST /api/v1/jobs/{job_id}/resume
POST /api/v1/jobs/{job_id}/cancel

GET /api/v1/jobs/{job_id}/metrics
  - Real-time training metrics (SSE stream)

GET /api/v1/jobs/{job_id}/samples
  - Generated samples during training

POST /api/v1/models/evaluate
  - Run evaluation benchmarks

GET /api/v1/models/compare
  - Compare multiple models

POST /api/v1/datasets/upload
  - Upload custom training data
```

### 5.4 CLI Tool

```bash
gold-cli create \
  --teacher meta-llama/Llama-3.1-70B \
  --student meta-llama/Llama-3.2-8B \
  --strategy on_policy \
  --lambda 0.5 \
  --dataset ultrachat_200k \
  --gpus 4

gold-cli status <job_id>

gold-cli logs <job_id> --follow

gold-cli evaluate \
  --model ./checkpoints/step-1000 \
  --benchmarks mmlu,gsm8k,humaneval

gold-cli export <job_id> \
  --format hf \
  --push-to-hub username/my-distilled-model
```

---

## 6. Technical Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────┐
│              Frontend (React)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Dashboard │  │Job Wizard│  │Analytics │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                      │ REST API
┌─────────────────────────────────────────────────┐
│           Backend (FastAPI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Job Queue │  │Metrics   │  │Model     │     │
│  │Manager   │  │Service   │  │Registry  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────┐
│         Training Workers (Ray/Slurm)            │
│  ┌──────────────────────────────────┐          │
│  │  GOLD Distillation Engine        │          │
│  │  - Teacher model inference       │          │
│  │  - Student training loop         │          │
│  │  - Metrics collection            │          │
│  └──────────────────────────────────┘          │
└─────────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────┐
│        Storage & Infrastructure                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │PostgreSQL│  │S3/Blob   │  │Redis     │     │
│  │(metadata)│  │(models)  │  │(cache)   │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

### 6.2 Tech Stack

**Frontend**
- React 18 + TypeScript
- Vite (build tool)
- TanStack Query (data fetching)
- Recharts (visualizations)
- Tailwind CSS + shadcn/ui (styling)
- Zustand (state management)

**Backend**
- FastAPI (Python 3.11+)
- SQLAlchemy (ORM)
- Celery or Ray (distributed task queue)
- Pydantic (validation)
- HuggingFace Transformers + TRL

**Infrastructure**
- Docker + Docker Compose
- PostgreSQL 15
- Redis 7
- S3-compatible storage (MinIO for local dev)
- Prometheus + Grafana (monitoring)

**Training**
- PyTorch 2.4+
- HuggingFace TRL (GOLD trainer)
- vLLM (teacher inference acceleration)
- DeepSpeed or FSDP (distributed training)

### 6.3 Data Models

```python
# Job Configuration
class DistillationJob:
    id: UUID
    user_id: str
    teacher_model: str
    student_model: str
    strategy: DistillationStrategy  # on_policy, off_policy, hybrid
    hyperparameters: dict
    dataset_config: dict
    hardware_config: dict
    status: JobStatus  # queued, running, paused, completed, failed
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

# Training Metrics
class TrainingMetrics:
    job_id: UUID
    step: int
    epoch: float
    loss: float
    teacher_loss: Optional[float]
    student_loss: float
    kl_divergence: float
    accuracy: float
    learning_rate: float
    tokens_per_second: float
    gpu_memory_used: float
    estimated_cost: float
    timestamp: datetime

# Model Evaluation
class EvaluationResult:
    job_id: UUID
    checkpoint: str
    benchmark: str  # mmlu, gsm8k, humaneval, etc.
    score: float
    details: dict
    timestamp: datetime
```

---

## 7. User Experience

### 7.1 Key User Flows

**Flow 1: First-time user creates distillation job**
1. Land on homepage → See hero section with value prop
2. Click "Create New Job" → Guided wizard
3. Select teacher (suggestions based on use case)
4. Select student (auto-suggest compatible models)
5. Choose distillation strategy (visual cards with pros/cons)
6. Review cost estimate and click "Launch"
7. Redirected to training dashboard

**Flow 2: Monitor training progress**
1. Dashboard shows real-time loss curves
2. Sample generations update every N steps
3. Receive Slack notification when job completes
4. Review evaluation benchmarks
5. Download model or push to HuggingFace Hub

**Flow 3: Compare multiple distillation runs**
1. Navigate to "Experiments" tab
2. Select multiple job IDs
3. View side-by-side comparison:
   - Loss curves overlaid
   - Evaluation scores table
   - Cost vs performance scatter plot
   - Generation quality samples
4. Export comparison report as PDF

### 7.2 UI/UX Principles

**Simplicity First**
- Sane defaults for all hyperparameters (one-click distillation)
- Progressive disclosure (advanced options hidden initially)
- Inline help text and tooltips

**Transparency**
- Real-time cost estimation
- Clear explanations of distillation strategies
- Sample generations to assess quality visually

**Speed**
- Optimistic UI updates
- Server-sent events for real-time metrics
- Lazy loading for heavy data

**Accessibility**
- WCAG 2.1 AA compliance
- Keyboard navigation
- Screen reader support

---

## 8. Success Metrics

### 8.1 Product Metrics

**Adoption**
- Weekly active users
- Jobs created per week
- User retention (7-day, 30-day)

**Engagement**
- Average jobs per user
- Time spent on training dashboard
- Model downloads/exports

**Quality**
- Average distillation quality (eval benchmark scores)
- Cost reduction achieved (vs baseline inference)
- Job success rate

### 8.2 Technical Metrics

**Performance**
- API response time (p50, p95, p99)
- Dashboard load time
- Training throughput (tokens/sec)

**Reliability**
- Uptime (99.9% target)
- Job failure rate (<5%)
- Data loss incidents (zero target)

**Efficiency**
- GPU utilization (>85%)
- Cost per distillation job
- Storage costs

---

## 9. Development Phases

### Phase 1: MVP (8 weeks)
**Goal**: Basic on-policy distillation with simple UI

**Features**
- [x] Core distillation engine (on-policy only)
- [x] Basic web UI (job creation + monitoring)
- [x] Single-node training (1-4 GPUs)
- [x] PostgreSQL persistence
- [x] HuggingFace model integration
- [x] Basic evaluation (MMLU, GSM8K)

**Success Criteria**
- 10 beta users running successful distillation jobs
- Average 70%+ retention of teacher model quality
- 5-10x cost reduction vs teacher inference

### Phase 2: Enhanced Features (6 weeks)
**Goal**: Production-ready with advanced features

**Features**
- [ ] Hybrid distillation (on-policy + off-policy)
- [ ] Cross-tokenizer support (ULD)
- [ ] vLLM teacher acceleration
- [ ] Multi-node distributed training
- [ ] Advanced UI (comparison view, A/B testing)
- [ ] CLI tool
- [ ] Webhook notifications
- [ ] Model registry and versioning

**Success Criteria**
- 100+ active users
- Support for 10+ model families
- <2% job failure rate

### Phase 3: Scale & Optimize (6 weeks)
**Goal**: Enterprise-ready platform

**Features**
- [ ] API-based teachers (OpenAI, Anthropic)
- [ ] Custom evaluation benchmarks
- [ ] Team collaboration features
- [ ] Role-based access control
- [ ] Cost budgets and alerts
- [ ] Auto-scaling infrastructure
- [ ] Marketplace for distilled models
- [ ] White-label deployment option

**Success Criteria**
- 500+ users
- 1000+ successful distillations
- 3 enterprise customers

### Phase 4: Research & Innovation (Ongoing)
**Goal**: Push state-of-the-art in distillation

**Features**
- [ ] Multi-teacher distillation
- [ ] Curriculum learning strategies
- [ ] Automated hyperparameter tuning
- [ ] Distillation quality predictor (before training)
- [ ] Novel distillation algorithms
- [ ] Research partnerships and paper publications

---

## 10. Technical Specifications

### 10.1 API Design

**REST Conventions**
- Follow RESTful principles
- Versioned URLs (`/api/v1/...`)
- Standard HTTP status codes
- JSON request/response bodies
- JWT authentication

**Rate Limiting**
- 100 requests/minute per user (authenticated)
- 10 requests/minute (unauthenticated)
- Upgradeable tiers for heavy users

**Error Handling**
```json
{
  "error": {
    "code": "INVALID_MODEL",
    "message": "Teacher model not found",
    "details": {
      "model_id": "invalid-model-name",
      "suggestions": ["meta-llama/Llama-3.1-70B"]
    }
  }
}
```

### 10.2 Security

**Authentication & Authorization**
- OAuth 2.0 + JWT tokens
- HuggingFace SSO integration
- API keys for programmatic access
- Row-level security (users see only their jobs)

**Data Protection**
- Encryption at rest (S3, PostgreSQL)
- Encryption in transit (TLS 1.3)
- Model weights stored with access controls
- GDPR compliance (data export/deletion)

**Infrastructure Security**
- VPC isolation for training workers
- Secrets management (Vault/AWS Secrets Manager)
- Regular security audits
- Dependency scanning (Dependabot)

### 10.3 Scalability

**Horizontal Scaling**
- Stateless API servers (scale with load balancer)
- Distributed training workers (Ray cluster)
- Read replicas for PostgreSQL
- CDN for static assets

**Vertical Scaling**
- Support for A100 80GB, H100 GPUs
- Multi-node training (DeepSpeed, FSDP)
- Model parallelism for large teachers

**Optimization**
- Redis caching for model metadata
- Connection pooling
- Lazy loading of model weights
- Gradient checkpointing for memory efficiency

---

## 11. Risks & Mitigations

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GPU availability | High | Medium | Queue management, spot instances, multi-cloud |
| Model quality variance | High | Medium | Extensive testing, quality gates, auto-rollback |
| Training instability | Medium | Medium | Checkpoint frequently, gradient clipping |
| Storage costs | Medium | High | Lifecycle policies, compression, deduplication |

### Product Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Low user adoption | High | Medium | Beta program, user research, marketing |
| Competitors (HF, OpenAI) | High | Medium | Focus on UX, unique features, community |
| Unclear value prop | Medium | Low | Case studies, ROI calculator, testimonials |
| Pricing complexity | Low | High | Simple tiered pricing, usage calculator |

### Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Runaway costs | High | Low | Budget alerts, auto-shutdown, resource limits |
| Data loss | High | Low | Automated backups, replication, disaster recovery |
| Legal (model licensing) | Medium | Low | Terms of service, user responsibility clause |

---

## 12. Go-to-Market Strategy

### 12.1 Target Markets

**Primary**: AI/ML teams at tech companies
- 50-500 employees
- Already using LLMs in production
- Pain: High inference costs

**Secondary**: AI research labs
- Universities and research institutions
- Need: Reproducible distillation experiments

**Tertiary**: Individual developers
- Building AI products with constrained budgets
- Need: Easy, affordable model compression

### 12.2 Pricing Model

**Free Tier**
- 10 hours GPU time/month
- Public models only
- Community support

**Pro Tier** ($99/month)
- 100 hours GPU time/month
- Priority queue
- Private models
- Email support

**Enterprise Tier** (Custom)
- Unlimited GPU time
- Dedicated infrastructure
- White-label deployment
- SLA + premium support
- Custom integrations

### 12.3 Launch Plan

**Pre-launch (2 months before MVP)**
1. Build waitlist landing page
2. Publish blog posts on OPD benefits
3. Engage with ML community on Twitter/LinkedIn
4. Partner with AI newsletters

**Launch (MVP ready)**
1. ProductHunt launch
2. HackerNews post with technical deep-dive
3. Demo video + tutorial
4. HuggingFace Spaces demo

**Post-launch**
1. User interviews and feedback loops
2. Case studies with early adopters
3. Conference talks (NeurIPS, ICML workshops)
4. Integration partnerships (LangChain, LlamaIndex)

---

## 13. Appendix

### 13.1 Glossary

- **On-Policy Distillation**: Student model generates outputs, teacher provides feedback
- **Off-Policy Distillation**: Student learns from teacher-generated or human-generated data
- **GKD (Generalized Knowledge Distillation)**: Framework that interpolates between on-policy and off-policy
- **ULD (Universal Logit Distillation)**: Cross-tokenizer distillation technique
- **Lambda (λ)**: Ratio of on-policy to off-policy samples
- **Beta (β)**: Interpolation parameter in generalized JSD

### 13.2 References

- "On-Policy Distillation of Language Models" (Agarwal et al., 2023)
- "Generalized Knowledge Distillation" (HuggingFace TRL docs)
- "Universal Logit Distillation" (for cross-tokenizer KD)
- HuggingFace TRL GOLDTrainer implementation

### 13.3 Open Questions

1. **Multi-modal distillation**: Should we support vision-language models?
2. **Continuous learning**: Enable incremental distillation as teacher improves?
3. **Federated distillation**: Allow distributed training across user data?
4. **Distillation marketplace**: Platform for sharing/selling distilled models?

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-04 | GOLD Team | Initial PRD |

---

**Next Steps:**
1. Review PRD with stakeholders
2. Create detailed technical design docs
3. Set up project repository and development environment
4. Begin Phase 1 development sprint planning

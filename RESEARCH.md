# **Model Forensics: Causal RCA for Foundation Models**

# Root Cause Analysis (RCA) for Frozen Models

## A Model-Centric, Causality-Based Framework for Explaining Specific Failures

---

## Part 1 — Academic Research Proposal

### Motivation

Foundation models are increasingly used in critical domains, but we lack a robust methodology for **post-hoc causal analysis** of specific incidents without retraining. This project aims to create a reproducible framework to uncover the internal causal chain behind failures in *frozen* models, delivering both scientific insight and operational incident response.

### Research Questions

1. **Minimality:** Identify the smallest causal feature set that reverses the failure without harming unrelated performance.
2. **Interaction dynamics:** Determine whether failures are caused by unique feature combinations.
3. **Structural anomaly detection:** Test if compressibility/entropy can locate anomalous computation paths.
4. **Trajectory mapping:** Find the earliest irreversible decision point in the forward pass.
5. **Training provenance:** Trace failures back to training data or update history.

### Methodology — Deep Dives by Module

#### 1. Causal Minimal Fix Set (CCA)

* **Theoretical:** Pearlian do-operations to find minimal sufficiency/necessity sets.
* **Practical:** Greedy/beam search over top-K activations; interventions by zeroing/patching.
* **Concerns:** High cost, risk of confounding; **Mitigation:** progressive narrowing, early-stop heuristics.

#### 2. Interaction Spectroscopy (IS-C)

* **Theoretical:** Shapley-Taylor indices and joint mutual info for multi-feature causation.
* **Practical:** Enumerates/test pairs/triads; visualizes causal graphs.
* **Concerns:** Scaling issues; **Mitigation:** sampling-based search, statistical pruning.

#### 3. Compression Forensics (CF)

* **Theoretical:** Uses compressibility gaps as anomaly signals.
* **Practical:** Layer-wise compression vs benign baseline; flags for deeper causal probing.
* **Concerns:** May not generalize; **Mitigation:** multi-architecture baselines, fallback heuristics.

#### 4. Decision Basin Cartography (DBC)

* **Theoretical:** Finds earliest irreversible activation basin.
* **Practical:** Layer-by-layer perturbation; outputs earliest irreversible point.
* **Concerns:** Dense sweeps required; **Mitigation:** guided perturbations using CF/CCA hints.

#### 5. Training-Time Provenance

* **Theoretical:** Influence functions + gradient attribution.
* **Practical:** Links features to specific training data; fallback: synthetic probing via counterfactual inputs designed to stress suspect features.
* **Concerns:** Influence instability, cost; **Mitigation:** scalable approximations, well-defined synthetic probing.

### Evaluation Protocols & Experimental Design

* **Architectures & Modalities:** LLMs, ViTs/CNNs, multimodal.
* **Incidents & Suites:** Diverse, reproducible failure cases.
* **Statistical Testing:** Paired designs, permutation tests, FDR control.
* **Power & Budgets:** Runtime SLAs per module.
* **Cross-Checkpoint Stability:** Fix-set stability across training stages.
* **Adversarial Controls:** Random/sham feature tests.
* **Module-Specific Benchmarks:** Success metric per module (e.g., DBC ≥90% earliest point consistency).

### Theoretical Contributions

* Extends causality to frozen models.
* Links anomaly detection and causal validation.
* Formalizes decision basin mapping.
* Introduces scalable multi-feature interaction measures.

### Example Scenarios

* LLM jailbreak.
* Rare medical misclassification.
* Multimodal agent API misfire.

---

## Part 2 — Practical Proposal

### Goal

Deliver a **ready-to-use RCA toolkit** producing a Root Cause Certificate within hours of an incident.

### Core Components

* **RCA Recorder:** Selective activation capture.
* **RCA Player:** Deterministic replay.
* **CF Triager:** Anomaly detection.
* **CCA Search:** Minimal fix-set.
* **IS-C Engine:** Multi-feature causation.
* **DBC Mapper:** Earliest basin entry.
* **Provenance Engine:** Training linkage.
* **RCA Card Generator:** Standardized RCA schema.

### Integration Flow

Modules run independently or sequentially: CF → CCA → IS-C → DBC → Provenance → RCA Card.

### Deliverables

* Public “Incident Zoo”
* Modular toolkit with API/CLI
* Reproducible pipelines

### Success Criteria

* RCA ≤ 2 hours
* ≥80% causal flip rate
* ≤1% benign degradation
* Model-agnostic

### Cross-Architecture Validation & Ops Readiness

* Architecture × incident success matrix
* Portability tests
* Lightweight monitoring hooks
* Runtime SLAs

### Known Concerns & Mitigations

1. **Computational Complexity:** Progressive narrowing, sampling, distributed processing.
2. **Causal Validation:** Falsification tests, counterfactual consistency.
3. **Generalization Assumptions:** Benchmark widely, fallback heuristics.
4. **Training Provenance:** Scalable influence approximations, defined synthetic probing.

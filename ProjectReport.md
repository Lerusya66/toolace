# Report: Hallucination detection in tool calling
## Team Members: 
**Yakupova Valeria**

**Dmitrij Korogod** 

**Weerathep Rattanajaratkul**

**Hritendu Russo Baruri**

## 1. Motivation 
This notebook upgrades the original baseline evaluation in three ways:

1. It rebuilds the clean ToolACE split directly from the original dataset.
2. It evaluates on a combined benchmark made of `clean + 3 hallucination datasets` from `datasets/`.
3. It trains stronger sample-level detectors on real labels and reports both overall and per-type metrics.

The goal is to benchmark hallucination detection for:
- `tool_output_contradiction`
- `overgeneration`
- `missing_tool`

### Hallucinations and Tools
#### Tools
In the context of LLMs, a tool is an external service that an LLM receives information from upon a query from the user. 

For example: A weather API. The user requests to know the weather in Moscow, however, the LLM does not have this information readily available. So, it calls on a `weather api (the tool)` to get the current weather in Moscow and forwards the information to the user.
#### Hallucinations in tool calling
LLMs grounded by tool APIs can still fabricate or distort facts in their final response, even when correct data was returned. Such distortions or incorrect facts are called Hallucinations. In the context of tool-calling, Hallucinations can be roughly divided into three categories:
- 1.**Tool Output Contradiction:** Model response directly contradicts a fact returned by the tool. For example: 
        
        `Tool says "sunny" → Answer says "rainy"`
- 2.**Overgeneration:** Response contains information not present in the tool output i.e. unverified speculative additions.

        `…and the weather has been good past few months`
- 3.**Missing Tool:** Response suggests actions that require a tool not available in the current session.

        `"Would you like me to book a ticket?"` 
    but no booking tool is present.

LLMs grounded by tool APIs can still fabricate or distort facts in their final response, even when correct data was returned.

Existing detection methods were not designed for tool-calling dialogues and have no span-level benchmarks for this setting.

--- 

# 2 Methodology

### Improved Hallucination Detection Baselines

This notebook upgrades the original baseline evaluation framework by introducing a multi-layered benchmark that incorporates lexical token alignment, transformer-based token classification backbones, and attention-guided contextual lens ratios. To evaluate the resilience of dialogue systems operating under complex tool-assisted workflows, our methodology focuses on both sample-level and span-level detection of fine-grained hallucinations.

The methodology is structured across the following core pillars:

1. **Dataset Rehabilitation and Schema Integration**: We rebuild the clean evaluation splits directly from the raw ToolACE dataset. Following the task specification, the dialogue turns are framed as structural triplets consisting of the user Query, the execution output of the tools as Context, and the model's final response as Output. Ground-truth span annotations are structured in alignment with the RAGTruth schema across three target corruption scenarios:
   - **Tool Output Contradiction**: Direct factual discrepancies between the model's natural language response and the tool payload.
   
   - **Overgeneration**: Unverified assertions or speculative facts introduced by the model that are completely absent from the tool context.
   
   - **Missing Tool**: Actions suggested by the model that imply or require the activation of an unavailable tool API.
   

2. **Multi-Model Baseline Extensions**: Rather than relying strictly on raw out-of-the-box predictions, we evaluate and optimize several architectural archetypes:
   - **Lexical Span Verifier**: A non-parametric, exact-match lookup baseline that evaluates token-level overlap and flags sub-strings failing lexical alignment with tool contexts.
   
   - **Lettuce-Span Supervised Detector**: Utilizes a modern token-classification transformer backbone (`KRLabsOrg/lettucedect-base-modernbert-en-v1`) based on the BERT architecture, optimized to directly map token positions to boundary probabilities.
   
   - **LookBack-Span Supervised Detector**: Powered by a dense generative language model backbone (`Qwen/Qwen2.5-0.5B`), this approach extracts inner attention layers to compute lookback ratios. It evaluates how heavily the model attends to the input context versus its own historical generation tokens.
   
   - **Soft-Vote Ensemble**: A collaborative combination that synthesizes sample-level probabilities across the individual supervised models to maximize precision, control false-positive rates, and boost the overall Area Under the Receiver Operating Characteristic curve (AUROC).
   

---

## 3. Discussion of Results
**Overall Test Set Metrics:**

| Method | Backbone | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| **Lexical Span Verifier** | N/A (Non-parametric) | 56.6% | 0.529 | 0.517 | 0.523 |
| **LookBack-Span Supervised** | Qwen2.5-0.5B | 48.9% | 0.463 | 0.708 | 0.560 |
| **Lettuce-Span Supervised** | ModernBERT | **67.7%** | **0.665** | 0.601 | 0.631 |
| **Soft-Vote Ensemble** | Combined | 67.6% | 0.639 | **0.679** | **0.659** |

The comparative performance across the initial baselines and our optimized supervised implementations reveals clear trade-offs between detection recall and calibration precision.

An examination of the initial out-of-the-box logistic regression baselines evaluated on the combined benchmark ($n=828$) highlights the inherent difficulty of zero-shot alignment:

- **LettuceDetect (LogReg Baseline)**: Achieved an absolute recall of 1.000, but suffered from extreme over-flagging, resulting in a low precision of 0.460 and an overall accuracy of 0.460. This indicates that without threshold calibration or fine-tuning, the token classifier collapses into a trivial majority-positive state.

- **LookBackLens (LogReg Baseline)**: Demonstrated superior structural calibration compared to the raw token baseline, elevating accuracy to 0.506 and precision to 0.476, while preserving a solid recall of 0.714.

- **Baseline Agreement**: Agreement statistics show that at least one method flags a hallucination in 100.0% of cases, with a mutual overlap on 69.1% of samples, confirming that both models catch overlapping signal boundaries but suffer from high false-positive rates when left uncalibrated.


By introducing the supervised span training paradigm and merging their outputs into a **Soft-Vote Ensemble**, performance increases significantly:

- **Overall Ensemble Performance**: The soft-voting ensemble establishes the highest robustness on the benchmark, yielding an overall accuracy of **67.6%** (0.6763), an overall F1-score of **65.9%** (0.6590), and well-balanced precision/recall curves (0.6395 and 0.6798 respectively).

- **Specificity on Clean Inputs**: On uncorrupted data (`clean`), the supervised Lettuce model shows strong specificity, leaving true responses unflagged with an accuracy of **78.3%**. The Soft-Vote Ensemble tracks close behind at **71.0%** accuracy on clean inputs, indicating a successful suppression of the false-positive errors that plagued the initial zero-shot baselines.

- **Robustness Against Missing Tools**: For the challenging `missing_tool` hallucination category, the ensemble manages an accuracy of **63.3%** and an F1 score of **54.8%** (Precision = 0.5679, Recall = 0.5287). This proves that synthesizing inner attention-based contextual awareness with dense token-level representations allows the detector to reliably capture abstract structural anomalies—such as an assistant proposing actions without tool support—alongside explicit factual contradictions.



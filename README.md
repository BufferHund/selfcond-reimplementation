# Introduction
This report presents a toy project implementation of the self-conditioning mechanism for pre-trained language models (TLMs) as described in the paper "Self-Conditioning Pre-Trained Language Models" by Suau et al. (2022). The primary goal of this toy project is to demonstrate the core functionality of the self-conditioning approach in a simplified and accessible manner.


# Method
Self-Conditioning enables control over pretrained TLMs without additional parameters. It utilizes "expert units"—neurons sensitive to specific concepts—to guide text generation. The key steps include:
1. Identifying model units responsive to specific concepts.
2. Computing their expertise using AP metrics.
3. Activating these units during generation to steer output.



# Project Structure
Unlike the original package-based design, this reimplementation adopts a script-based structure:

| Script Name | Purpose |
|------------|---------|
| **compute_responses.py** | Collects model responses and saves hidden states in `.npy` format. |
| **compute_expertise.py** | Analyzes neuron expertise, computes AP scores, and generates CSV reports. |
| **generate_seq.py** | Generates text based on conditioned concepts, supporting various intervention strategies. |


# Usage Guide
## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/selfcond-reimplementation
cd selfcond-reimplementation
conda create --name selfcond python=3.8 -y
conda activate selfcond
pip install -r requirements.txt
```

## Execution Steps
### Step 1: Discovering Expert Units
```bash
python scripts/compute_responses.py \
--model-name "gpt2-medium" \
--data-path assets/football \
--save-path ./responses \
--batch-size 32 \
--max-length 128 \
--num-samples 1000
```
### Step 2: Analyzing Neuron Expertise
```bash
python scripts/compute_expertise.py \
--root-dir ./responses \
--assets-dir ./assets \
--model-name "gpt2-medium" \
--top-k 10 \
--show
```
### Step 3: Generating Conditioned Text
```bash
python scripts/generate_seq.py \
--model-name "gpt2-medium" \
--expertise "responses/sense/football-1_04_00__/gpt2-medium/expertise/expertise.csv" \
--length 20 \
--prompt "Once upon a time" \
--seed 0 1 2 3 4 5 6 7 8 9 \
--temperature 1.0 \
--metric ap \
--forcing "on_p50" \
--num-units 50 \
--no-save
```





# Comparison with Original Implementation
This toy project aims to provide a simplified and accessible implementation of the self-conditioning mechanism. The primary differences from the original implementation include:


| Aspect                | Apple's Implementation                     | Toy's Implementation                        |
|-----------------------|--------------------------------------------|-------------------------------------------|
| Input Style           | Pre-processed Dict[str, np.ndarray]        | Layer-by-layer loading from .npy files    |
| Storage Content       | Computed metrics only (AP, forcing values) | Computed metrics + layer statistics       |
| Data Retention        | Stores processed metrics only (AP, forcing values) | Loads raw responses from .npy files       |
| Memory Management     | Processes per-layer responses sequentially | Concatenates all layers upfront (higher memory) |
| Parallelization       | Multi-threaded AP calculation (Pool)       | Single-threaded processing                |
| Neuron Processing     | Vectorized layer-wise operations           | Per-neuron iteration (1024/layer)          |
| Model Assumptions     | Architecture-agnostic                      | GPT2-specific (24 layers, 1024 units)      |
| Device Support        | GPU acceleration                           | Forced CPU usage                          |
| User Interface        | Library-style API                          | Command-line interface with logging       |



# References
- Suau, Xavier, Luca Zappella, and Nicholas Apostoloff. "Self-Conditioning Pre-Trained Language Models." *International Conference on Machine Learning*, 2022.
- Hinton, Geoffrey E. "Products of experts." *ICANN*, 1999.
- *GitHub repository: Self-Conditioning Pre-Trained Language Models (ICML 2022).*





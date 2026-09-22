# Vir2vec: A Genome-Wide Viral Embedding

**Vir2vec** is a family of decoder-only genomic language models (17M, 138M, and 422M parameters) obtained via continual pretraining of [Mistral-DNA](https://huggingface.co/RaphaelMourad/Mistral-DNA-v1-422M-hg38) on a curated pan-viral corpus of **565,747 complete genomes across 295 viral species**. A single fixed-length 4,096-dimensional embedding per genome or read is produced by max-pooling the model's vocabulary logits, which downstream shallow classifiers use for tasks such as viral discrimination, host-range prediction, and variant typing.

Alongside the model, this repository hosts the **viral Genome Understanding Evaluation (vGUE)** benchmark — a dedicated evaluation suite probing viral representations across five biological dimensions: broad organism-level distinctions, genome-wide evolutionary signatures, intra-genus species separation, variant/subtype typing, and phenotypic context signal identification.

> Vir2vec: A Genome-Wide Viral Embedding. Rancati, S.*, Arozarena Donelli, P.*, et al. (2026). *bioRxiv*.

## Model & benchmark

| Resource | Link | Notes |
| :--- | :--- | :--- |
| Vir2vec model | [`pabloarozarenad/Vir2vec`](https://huggingface.co/pabloarozarenad/Vir2vec) | **Gated** (manual approval). `main` branch = 422M; `revision="138M"` / `revision="17M"` for smaller scales. |
| vGUE benchmark | [`pabloarozarenad/vGUE-benchmark`](https://huggingface.co/datasets/pabloarozarenad/vGUE-benchmark) | 7 task subsets, public. |

Baselines compared against in the paper: [Mistral-DNA](https://huggingface.co/RaphaelMourad/Mistral-DNA-v1-422M-hg38) (human-trained), [ModernBERT-DNA-virus](https://huggingface.co/RaphaelMourad/ModernBert-DNA-v1-37M-virus) (viral-specific), and Evo-1 (7B-parameter prokaryotic/phage model).

## Repository layout

```text
Vir2vec/
├── config/
│   └── default_config.yaml         # accelerate launch config (multi-GPU training)
├── data/
│   └── README.md                   # expected layout for pretraining sequence files
├── notebooks/
│   ├── 01_pooling_mean_max.ipynb        # Mean vs. max pooling strategy comparison
│   ├── 02_explainability_routing.ipynb  # ArgMax-routed SHAP -> single-codon attribution
│   ├── classifiers/                     # vGUE downstream classifiers (run after embeddings/)
│   │   ├── 01_classifier_logistic_regression.ipynb
│   │   ├── 02_classifier_random_forest.ipynb
│   │   ├── 03_classifier_xgboost.ipynb
│   │   └── 04_classifier_anomaly_detection.ipynb   # Virus-vs-bacteria one-class detection
│   └── embeddings/                      # Feature extraction across genomic LMs
│       ├── vir2vec_embs.ipynb
│       ├── mistraldna_embs.ipynb
│       └── modernbertvirus_embs.ipynb
├── scripts/
│   ├── train_vir2vec17m.py
│   ├── train_vir2vec138m.py
│   └── train_vir2vec422m.py
├── requirements.txt
└── README.md
```

Run order for the notebooks: `notebooks/embeddings/*.ipynb` write `.h5` embedding files first; `notebooks/classifiers/*.ipynb` then read those `.h5` files (default relative path `../embeddings/<task>_<model>.h5`).

## Quickstart

### 1. Environment

```bash
conda create -n vir2vec python=3.10 -y
conda activate vir2vec
pip install -r requirements.txt

# Optional, GPU nodes only:
pip install flash-attn --no-build-isolation
```

Authenticate to Hugging Face once (`pabloarozarenad/Vir2vec` is gated — request access on the model page first):

```bash
export HF_TOKEN="hf_..."
huggingface-cli login --token "$HF_TOKEN"
```

### 2. Compute a Vir2vec embedding

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "pabloarozarenad/Vir2vec"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

dna_sequence = "ACGTAGCATCGCGATGACTGCATCACT"
inputs = tokenizer(dna_sequence, return_tensors="pt")

with torch.no_grad():
    # Forward pass to obtain raw vocabulary logits [batch, seq_len, 4096]
    outputs = model(**inputs)
    logits = outputs.logits

    # Max-pool logits over sequence length -> 4,096-dimensional embedding
    embedding = torch.max(logits, dim=1).values[0]
```

### 3. Load a vGUE task

```python
from datasets import load_dataset

ds = load_dataset("pabloarozarenad/vGUE-benchmark", "dna_rna", split="test")
```

### 4. Continual pretraining

To run pretraining CLI execution or dry runs:

```bash
python scripts/train_vir2vec17m.py \
  --train_csv path/to/train.csv --val_csv path/to/val.csv \
  --batch_size 1 --max_steps 1
```

For a full multi-GPU run, launch through `accelerate`:

```bash
accelerate launch --config_file config/default_config.yaml scripts/train_vir2vec422m.py \
  --train_csv train_dataset.csv --val_csv val_split.csv
```

Training CSVs require a single `sequence` column; see `data/README.md` for how sequence files are organized.

## Citation

If you use Vir2vec or vGUE, please cite:

```bibtex
Rancati, S.*, Arozarena Donelli, P.*, Nicora, G., Bergomi, L., Buonocore, T.M., Sy, M.A.,
Pandey, S., Prosperi, M., Salemi, M., Bellazzi, R., Boucher, C., Parimbelli, E.+, Marini, S.+
Vir2vec: A Genome-Wide Viral Embedding. bioRxiv (2026).
```

## Contact

- Simone Marini — simone.marini@ufl.edu
- Enea Parimbelli — enea.parimbelli@unipv.it
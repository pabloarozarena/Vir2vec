# Dataset Directory Setup

Large sequence text files are excluded from Git tracking (see repo `.gitignore`: `data/train/`, `data/test/`, `data/validation/`, `*.fasta`, `*.txt`).

## Pretraining corpus

Vir2vec was continually pretrained on a curated pan-viral corpus of **565,747 complete genomes across 295 viral species**, compiled from:

- [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/)
- [BV-BRC](https://www.bv-brc.org/) (Bacterial and Viral Bioinformatics Resource Center)
- [GISAID](https://gisaid.org/)
- [LANL HIV Sequence Database](https://www.hiv.lanl.gov/) (LANL-HIVdb)
- [HBVdb](https://hbvdb.lyon.inserm.fr/)

Candidate sequences were filtered on genome completeness and missing-base thresholds, and hyper-dominant species (e.g. SARS-CoV-2, *Alphainfluenzavirus*) were down-sampled to reduce taxonomic sampling bias inherent to public repositories. The final corpus was split 70/30 into train/test by species-preserving proportions, with a small validation set drawn from the training split for hyperparameter tuning and early stopping. The held-out test partition is not used in pretraining and instead backs the vGUE benchmark.

## Expected layout for `scripts/train_vir2vec*.py`

The training scripts take flat CSV files (`--train_csv`, `--val_csv`) with a single `Sequence` column, e.g.:

```csv
Sequence
ACGTACGT...
TTGCATGC...
```

If you are staging raw per-source sequence files before building those CSVs, organize them as:

```text
data/
├── train/
│   ├── train__BV-BRC.txt
│   ├── train__GISAID.txt
│   ├── train__HBVdb.txt
│   ├── train__LANL-HIV-DB.txt
│   └── train__NCBI_virus.txt
├── validation/
│   ├── val__BV-BRC.txt
│   └── ...
└── test/
    ├── test__BV-BRC.txt
    └── ...
```

Each `.txt` file is one FASTA-derived nucleotide sequence per line, source-tagged by filename (`<split>__<repository>.txt`). Convert these into the `Sequence`-column CSVs the training scripts expect before running `scripts/train_vir2vec*.py`.

## vGUE benchmark data

The benchmark's 7 task subsets are hosted directly on Hugging Face — no local staging needed:

```python
from datasets import load_dataset
ds = load_dataset("pabloarozarenad/vGUE-benchmark", "dna_rna", split="test")
```

| Subset config | Purpose |
| :--- | :--- |
| `virus_bacteria` | Virus vs. non-virus (5kb bacterial contigs) |
| `metagenomic_reads` | Short-read (150bp) viral identification |
| `dna_rna` | DNA vs. RNA virus classification |
| `hiv1_hiv2` | HIV-1 vs. HIV-2 intra-genus separation |
| `sarscov2_subtyping` | SARS-CoV-2 lineage subtyping (7 clades) |
| `host_prediction` | Broad host-range classification |
| `hiv1_tropism` | HIV-1 tissue tropism (brain vs. non-brain) |

> **Known issue (as of this audit):** the dataset card's declared schema (`Sequence: string`, `label: int64`) does not match the actual parquet columns (`sequence: string`, `label: string`) for all 7 subsets, so `load_dataset()` currently raises a `CastError`. See `AUDIT_REPORT.md` in the working directory for the exact diagnosis and fix.

## Local `.h5` embedding files

`notebooks/embeddings/*.ipynb` write per-task embedding matrices to local HDF5 files (not tracked in Git — add `*.h5` to `.gitignore` if you keep them alongside the repo). `notebooks/classifiers/*.ipynb` read these back from a relative `../embeddings/<task>_<model>.h5` path; keep the naming consistent between the two notebook stages or adjust the `H5_PATH` constant in each classifier notebook.

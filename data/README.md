

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

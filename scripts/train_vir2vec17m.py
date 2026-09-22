# IMPORT LIBRARIES
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32 "

import torch
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

torch.backends.cudnn.benchmark = True

BASE_MODEL = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"


def parse_args():
    parser = argparse.ArgumentParser(description="Continual pretraining of Vir2vec-17M on viral genomes.")
    parser.add_argument("--train_csv", default="train_dataset.csv", help="CSV file with a 'Sequence' column for training.")
    parser.add_argument("--val_csv", default="val_split.csv", help="CSV file with a 'Sequence' column for validation.")
    parser.add_argument("--cache_dir", default="cache_directory", help="Cache directory for downloaded models/datasets.")
    parser.add_argument("--output_dir", default="./results/models")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=float, default=10)
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, stop after this many steps (overrides num_train_epochs). Use 1 for a smoke test.")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    attn_implementation = "flash_attention_2" if (HAS_FLASH_ATTN and torch.cuda.is_available()) else "eager"
    if not HAS_FLASH_ATTN:
        print("flash_attn not installed; falling back to attn_implementation='eager'.")

    config = AutoConfig.from_pretrained(BASE_MODEL, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation, trust_remote_code=True)

    # LOAD BPE LETTER TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    print(tokenizer)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {pytorch_total_params/1000**2:.1f}M parameters")

    # LOAD DATA
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset_text = load_dataset(
        "csv",
        data_files={
            "train": args.train_csv,
            "validation": args.val_csv,
        },
    )

    columns_to_keep = ["Sequence"]
    for split in dataset_text:
        cols_to_remove = [col for col in dataset_text[split].column_names if col not in columns_to_keep]
        dataset_text[split] = dataset_text[split].remove_columns(cols_to_remove)

    # TOKENIZE
    def tokenize_function(examples):
        return tokenizer(examples["Sequence"], padding="longest", truncation=True, return_tensors="pt")

    dataset = dataset_text.map(tokenize_function, batched=True, load_from_cache_file=False)

    print(dataset["train"])

    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",  # Make sure both are the same
        save_strategy="epoch",  # Make sure both are the same
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir="./logs",
        # load_best_model_at_end is temporarily disabled: Trainer's reload of the
        # "best" checkpoint does not go through this custom Mixtral-DNA remote-code
        # class, so it silently drops/reinitializes all block_sparse_moe expert
        # weights (confirmed on both CPU and a real B200 GPU -- see AUDIT_REPORT.md).
        # Re-enable once that reload path is fixed to preserve MoE weights.
        load_best_model_at_end=False,
        fp16=False,  # Keep fp16=False
        gradient_accumulation_steps=4,
        warmup_steps=1000,
        logging_steps=100,
        max_grad_norm=1.0,
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # lower eval_loss is better
        disable_tqdm=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Start a trainer...")
    trainer.train()


if __name__ == "__main__":
    main()

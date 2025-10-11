from datasets import load_dataset

# Load the WMT14 dataset for German ↔ English
dataset = load_dataset("wmt/wmt14", "de-en")

# Inspect the available splits
print(dataset)

# Access train / test / validation sets
print(dataset["train"][0])  # shows a sample pair

def save_split(dataset_split, prefix):
    with open(f"data//{prefix}.en", "w", encoding="utf-8") as f_en, \
         open(f"data//{prefix}.de", "w", encoding="utf-8") as f_de:
        for row in dataset_split:
            f_en.write(row["translation"]["en"] + "\n")
            f_de.write(row["translation"]["de"] + "\n")

save_split(dataset["train"], "train")
save_split(dataset["validation"], "dev")
save_split(dataset["test"], "test")
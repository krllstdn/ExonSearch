import pickle

import h5py
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from utils import parse_gtf, preprocess_data

window_sizes = [101, 201, 301]
chromosome_names = [i for i in range(1, 23)] + ["X", "Y"]
RANDOM_SEED = 42

print("loading gtf...")
gene_groups = parse_gtf("data/Homo_sapiens.GRCh38.113.chr.gtf.gz")
# gene_groups = parse_gtf("MusMusculus/Mus_musculus.GRCm39.113.chr.gtf.gz")

print("splitting_genes...")
gene_ids = list(gene_groups.groups.keys())
train_genes, test_genes = train_test_split(
    gene_ids, test_size=0.2, random_state=RANDOM_SEED
)

with open("genes.pkl", "wb") as f:
    pickle.dump(
        {
            "gene_groups": gene_groups,
            "train_genes": train_genes,
            "test_genes": test_genes,
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

for window_size in window_sizes:
    for i in chromosome_names:
        print(f"Loading data for chromosome {i}....")
        # chromosomes = SeqIO.to_dict(SeqIO.parse(f'MusMusculus/Mus_musculus.GRCm39.dna.chromosome.{i}.fa', 'fasta'))
        chromosomes = SeqIO.to_dict(
            SeqIO.parse(f"data/Homo_sapiens.GRCh38.dna.chromosome.{i}.fa", "fasta")
        )

        # preprocess train/test data
        print("preprocessing train/test data.....")
        X_train, y_train = preprocess_data(
            train_genes, gene_groups, chromosomes, window_size
        )
        X_test, y_test = preprocess_data(
            test_genes, gene_groups, chromosomes, window_size
        )

        # save train/test data with h5
        with h5py.File(
            f"MusMusculus/window_{window_size}/train_test_data{i}.h5", "w"
        ) as f:
            # Save the tuple as a dataset
            f.create_dataset("X_train", data=X_train)
            f.create_dataset("y_train", data=y_train)
            f.create_dataset("X_test", data=X_test)
            f.create_dataset("y_test", data=y_test)

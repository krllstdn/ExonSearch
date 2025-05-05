import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

WINDOW_SIZE = 101
CHROMOSOME_PREFIX = ""
RANDOM_SEED = 42
LEARNING_RATE = 0.0005
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def preprocess_data(gene_ids, gene_groups, chromosomes, window_size):
    """Preprocess all data into memory before training."""
    X_data, y_data = [], []
    for gene_id in tqdm(gene_ids):
        try:
            gene_data = gene_groups.get_group(gene_id)
            seqs, lbls = generate_samples_for_gene(gene_data, chromosomes, window_size)
            if seqs:
                X_data.extend(seqs)
                y_data.extend(lbls)
        except KeyError:
            continue

    X_data = np.array(X_data, dtype=np.uint8)
    y_data = np.array(y_data, dtype=np.uint8)

    return X_data, y_data


def parse_gtf(gtf_path):
    """Parse GTF file and return exon positions grouped by gene"""
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "chr",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attributes",
        ],
    )

    # Filter exons and extract gene IDs
    exons = gtf[gtf["feature"] == "exon"].copy()
    exons["gene_id"] = exons["attributes"].str.extract(r'gene_id "([^"]+)"')
    exons["chr"] = CHROMOSOME_PREFIX + exons["chr"].astype(str)

    return exons.groupby("gene_id")


def generate_samples_for_gene(gene_data, chromosomes, window_size=101):
    sequences, labels = [], []
    half_window = window_size // 2
    chr_name = gene_data["chr"].iloc[0]

    if chr_name not in chromosomes:
        return [], []

    chr_seq = str(chromosomes[chr_name].seq)
    exon_starts = set(gene_data["start"].values)
    gene_start = gene_data["start"].min()
    gene_end = gene_data["end"].max()

    # Positive samples
    for start_pos in exon_starts:
        window_start = start_pos - half_window - 1
        window_end = window_start + window_size

        if 0 <= window_start < len(chr_seq) and window_end <= len(chr_seq):
            seq = chr_seq[window_start:window_end]
            sequences.append(one_hot_encode(seq))
            labels.append(1)

    # Negative samples
    num_neg = len(exon_starts)
    attempts = 0
    max_attempts = num_neg * 10

    while len(labels) - sum(labels) < num_neg and attempts < max_attempts:
        rand_pos = np.random.randint(gene_start, gene_end)
        if rand_pos not in exon_starts:
            window_start = rand_pos - half_window - 1
            window_end = window_start + window_size

            if 0 <= window_start < len(chr_seq) and window_end <= len(chr_seq):
                seq = chr_seq[window_start:window_end]
                sequences.append(one_hot_encode(seq))
                labels.append(0)
        attempts += 1

    return sequences, labels


def one_hot_encode(seq):
    mapping = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }
    return [mapping.get(nuc, [0, 0, 0, 0]) for nuc in seq]


class ChromosomeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, chr_files, batch_size=256):
        self.chr_files = chr_files
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(
                sum(h5py.File(f, "r")["X_train"].shape[0] for f in self.chr_files)
                / self.batch_size
            )
        )

    def __getitem__(self, index):
        # Get random chromosome
        chr_file = np.random.choice(self.chr_files)

        with h5py.File(chr_file, "r") as f:
            X = f["X_train"]
            y = f["y_train"]

            # Get random batch from chromosome
            start = np.random.randint(0, len(X) - self.batch_size)
            return (
                X[start : start + self.batch_size],
                y[start : start + self.batch_size],
            )

    def on_epoch_end(self):
        np.random.shuffle(self.chr_files)


class ValidationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, chr_files, batch_size=512):
        self.chr_files = chr_files
        self.batch_size = batch_size

    def __len__(self):
        total_samples = sum(
            h5py.File(f, "r")["X_test"].shape[0] for f in self.chr_files
        )
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, index):
        # Load one chromosome at a time
        chr_file = self.chr_files[index % len(self.chr_files)]

        with h5py.File(chr_file, "r") as f:
            X = f["X_test"][:]
            y = f["y_test"][:]

            return X, y

# 🧬 Exon Start Prediction in Human Genome Using Deep Learning
This project implements deep learning models for predicting exon start sites in the Homo sapiens genome using annotated reference data from  [Ensembl](https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/) (GRCh38). It features a custom pipeline for parsing GTF annotations, extracting sequence windows from reference FASTA files, and training convolutional neural networks to distinguish exon starts from other intragenic positions.

### Highlights
- **Dataset**: One-hot encoded 101/201/301bp windows from chromosomes 1–22, X, Y
- **Models**: Enhanced CNN with SE blocks, attention, inception-style convs
- **Performance**: Best model achieves 78.3% accuracy on a balanced validation set


### Important Files
- `preprocess_data.py` - script for data preprocessing. It creates the dataset for all chromosomes for the given `window_size`.``
- `utils.py` - utility functions for working with data
- `train.py` - script for model training.
- `models.py`  - file with model definitions
- `script.sh` - script for downloading data and running the code

More info can be found in `Exon Report.pdf`.
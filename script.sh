
# Download data
mkdir data
cd data

curl https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz

base_url="https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/"

for chr in {1..19} X Y; do
    wget "${base_url}/Homo_sapiens.GRCh38.dna.chromosome.${chr}.fa.gz"
done

# Unzip
gzip -dk *.gz


# generate dataset
mdir window_101
mdir window_201
mdir window_301

cd ..

# Setup environment
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

python3 preprocess_data.py

# train
python3 train.py

python3 compare.py



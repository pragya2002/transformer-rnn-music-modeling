# Transformer & RNN Scaling for Symbolic Music Modeling

This repository contains the code and experiments for my CS-GY 6923 optional project:

> **Scaling Laws for Transformers and RNNs on Symbolic Music (ABC Notation)**  
> *Submitted by: Pragya Awasthi (pa2755)*

The project investigates how decoder-only Transformers and LSTM-based RNNs scale when trained on a large symbolic music corpus derived from the **Lakh MIDI Dataset (LMD-full)**.  
I compare **five Transformer sizes** and **five RNN sizes** (≈1M → 100M+ parameters), fit power-law scaling curves, and train a large Transformer as the “best model” for music generation.

---

## 📁 Repository Structure

```text
transformer-rnn-music-modeling/
├── data_preprocessing/
│   ├── scripts/
│   │   ├── convert_midi_to_abc.py
│   │   ├── build_corpus.py
│   │   ├── build_vocab.py
│   │   ├── tokenize_corpus.py
│   │   ├── split_data.py
│   │   └── sequence_length_stats.py
│   └── data/        # NOT in repo (too large) – see below
├── notebooks/
│   ├── music_generation_project.ipynb
│   └── sample_generation_and_midi_conversion.ipynb
├── models/          # (optional) model definitions / helpers
├── configs/         # (optional) YAML/JSON configs if you add them later
├── requirements.txt
└── README.md
```

## 🎵 Data & Preprocessing

### 1. Dataset
Primary source: Lakh MIDI Dataset (LMD-full)  
#MIDI files: 176,581  
Converted to ABC notation, cleaned, and tokenized into a large corpus.  

Because the raw data and processed arrays are too large for GitHub, they are not included in this repo. Instead, I provide:

- Google drive link with compressed data (in `data_preprocessing/data/readme.md`)  

### 2. Data Folder (Not Tracked in Git)
Locally, the preprocessing pipeline assumes a layout like:

```text
data_preprocessing/
└── data/
    ├── midi_raw/            # Original MIDI files (from Lakh MIDI)
    ├── abc/                 # Converted ABC files (from convert_midi_to_abc.py)
    ├── corpus/
    │   └── abc_corpus.txt   # Concatenated cleaned ABC corpus
    └── tokenized/
        ├── vocab.json       # stoi / itos character vocabulary
        ├── all_ids.npy      # Long 1D array of token IDs
        ├── train_ids.npy
        ├── val_ids.npy
        └── test_ids.npy
```

### 3. Preprocessing Pipeline
Run these scripts locally, after downloading LMD:

```bash
# 1) Convert MIDI → ABC
python data_preprocessing/scripts/convert_midi_to_abc.py

# 2) Build cleaned ABC corpus
python data_preprocessing/scripts/build_corpus.py

# 3) Build character-level vocabulary
python data_preprocessing/scripts/build_vocab.py

# 4) Tokenize corpus to integer IDs (subset to 150M tokens)
python data_preprocessing/scripts/tokenize_corpus.py

# 5) Split into train/val/test
python data_preprocessing/scripts/split_data.py

# 6) (Optional) Compute sequence length statistics
python data_preprocessing/scripts/sequence_length_stats.py
```

These produce:

- `abc_corpus.txt` (≈3.56 GB)  
- `vocab.json` (100 unique characters)  
- `train_ids.npy` (100M tokens), `val_ids.npy` (25M), `test_ids.npy` (25M)

---

## 🔧 Environment Setup

```bash
git clone https://github.com/pragya2002/transformer-rnn-music-modeling.git
cd transformer-rnn-music-modeling

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies include: `torch`, `numpy`, `music21`, `tqdm`, `matplotlib`, `ipykernel` / `jupyter`

---

## 🧪 Experiments in Colab (Training & Scaling)

All core model training, scaling studies, and evaluation are done inside Google Colab notebooks.

### 1. Main Training & Scaling Notebook
Open: `notebooks/music_generation_project.ipynb` in Colab (or Jupyter with GPU).  

This notebook contains:

- Loading the `train_ids.npy`, `val_ids.npy`, `test_ids.npy` arrays from Drive  
- Transformer family training (Tiny → XL)  
- RNN/LSTM family training (Tiny → XL)  
- Logging:
  - Train loss per step
  - Validation loss per model size (after 1 epoch)
  - Parameter counts, epoch times, peak GPU memory
- Fitting power-law scaling curves:
  - Transformer: L(N) = a ⋅ N^-α + c  
  - RNN: same form, separate fit
- Generating plots:
  - Loss vs. step
  - Validation loss vs. parameter count (log-log) with power-law fit
- Loads the best XL Transformer checkpoint (`best_xl_weights.pt`)
- Computes final validation and test loss + perplexity  

You can run it cell-by-cell to reproduce Transformer and RNN scaling studies and comparisons.

### 2. Best Model & Sample Generation Notebook
Open: `notebooks/sample_generation_and_midi_conversion.ipynb`

This notebook:

- Generates:
  - Unconditional ABC samples
  - Conditional ABC samples (e.g., Happy Birthday / Jingle Bells)
- Cleans ABC, adds headers, and validates syntactically using `abc2midi` or `music21`  
- Converts to MIDI and saves to Google Drive  
- Computes:
  - Percentage of syntactically valid outputs  
  - Percentage that successfully convert to MIDI

---

## 🚀 How to Reproduce the Experiments

### A. Data (Local + Drive)
Run preprocessing scripts locally to create `train_ids.npy`, `val_ids.npy`, `test_ids.npy`, `vocab.json`.  
Upload these to Google Drive in the paths expected by notebooks (or adjust paths in first cells).

### B. Open Notebooks in Colab
- File → Open Notebook → GitHub or Google Drive  
- Open `music_generation_project.ipynb`  
- Set runtime to GPU (A100 / T4 / L4)  
- Edit paths to point to Drive locations for:
  - `train_ids.npy`, `val_ids.npy`, `test_ids.npy`  
  - `vocab.json`  
  - Checkpoint save directory  
- Run all cells to train all model sizes and gather scaling results  
- Run all cells in `sample_generation_and_midi_conversion.ipynb` to:
  - Load XL Transformer  
  - Generate unconditional/conditional samples  
  - Convert to MIDI  
  - Compute validity statistics

---

## 📈 Outputs & Artifacts

The notebooks produce:

- **Scaling plots:** validation loss vs. number of parameters (log-log)  
- **Training curves:** Loss vs. steps for each model  
- **Tables:** Model sizes, validation loss, perplexity, epoch time, GPU memory  
- **Music samples:** Unconditional and conditional ABC + MIDI.
  - Uncoditional sample midis: https://drive.google.com/drive/folders/1rpiCVmAnWQXj4OFQjhV_SyqyNl72eA3D?usp=drive_link
  - Conditional sample midis: https://drive.google.com/drive/folders/1joxc-ZyXSIKtLoprsWPFoqgGmt7Mj5sO?usp=drive_link

---

## 🧩 What Was Borrowed vs. Implemented

**Borrowed / adapted:**

- GPT-style architecture and training patterns from nanoGPT  
- AdamW, cosine LR schedule, warmup, gradient clipping  
- GPTConfig / GPTModel block structures

**Implemented / customized:**

- Full Lakh MIDI → ABC → cleaned corpus → tokenized splits pipeline  
- Character-level vocabulary & tokenization scripts  
- Experimental setup: 5 Transformer sizes, 5 RNN sizes  
- Logging, parameter counting, scaling-law fitting (in notebooks)  
- Best-model training strategy under Colab restrictions  
- ABC cleaning, header injection, integration with `abc2midi` / `music21`  
- Conditional/unconditional sampling workflows  
- MIDI export + validity statistics

---

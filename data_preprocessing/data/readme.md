📁 Data Folder

This folder contains (or links to) the preprocessed symbolic music dataset used for all Transformer and RNN experiments in this project.
The full dataset is too large for GitHub, so it is hosted on Google Drive.

📥 Download Dataset (Required)

🔗 Preprocessed ABC + Tokenized Dataset (Google Drive) in compressed form
https://drive.google.com/drive/folders/11i9D8xs85fGnUEKKNWVCvfL6wzyUrAi3?usp=sharing

Download and extract this into the data/ directory of the repository.

📦 Contents (after extraction)
```text
data/
├── raw_midi/              # (NOT in repo) original Lakh MIDI files (.mid)
│   └── ...                # 176k MIDI files, ~1.6 GB compressed / larger extracted
│
├── abc/                   # (NOT in repo) MIDI converted to ABC
│   ├── 000001.abc
│   ├── 000002.abc
│   ├── ...
│   └── 175926.abc
│
├── corpus/
│   └── abc_corpus.txt     # large concatenated ABC text corpus (~3.6 GB)
│
└── tokenized/
    ├── vocab.json         # character-level vocabulary (stoi / itos, 100 chars)
    ├── all_ids.npy        # int32 token sequence for 150M characters
    ├── train_ids.npy      # 100M tokens
    ├── val_ids.npy        # 25M tokens
    └── test_ids.npy       # 25M tokens
```
📝 Notes

Dataset originates from Lakh MIDI (176,581 files), converted to ABC.

Cleaned and processed using the scripts in data_preprocessing/.

Used for all training, scaling, and generation experiments.

# ParsSugg: Persian Suggestion Classification

Code and datasets for the paper **"A semi-supervised method to generate a Persian dataset for suggestion classification"** (Safari & Mohammady, *Language Resources and Evaluation*, 2024).

This is the first work on suggestion classification in Persian. Because no labeled Persian data existed, we propose a two-step semi-supervised method to build a dataset (**ParsSugg**) from hotel reviews, then evaluate five classifiers on it. The best model, fine-tuned **ParsBERT**, reaches an **F-score of 97.27** on the suggestion class.

## Overview

*Suggestion classification* decides whether a sentence contains an explicit suggestion, such as advice to other customers or a request for product improvement.

> "If you go to the Diamond Hotel, be sure to try the restaurants on Ferdowsi Street" → **suggestion**
> "There are quality and cheap restaurants on Ferdowsi Street" → **non-suggestion**

The dataset is generated in two steps:

1. **Manual labeling.** About 100 student annotators labeled 2,000 hotel-review sentences through a custom annotation website. Sentences with at least 60% positive votes went to two expert annotators (Cohen's Kappa = 0.85). The positive class was then augmented with about 1,000 translated and expert-reviewed English suggestion sentences, giving a balanced **Gold** dataset.
2. **Automatic extension.** The best classifier on the Gold data (ParsBERT) labeled 65,208 unlabeled sentences from several Iranian hotel-booking sites. After under-sampling the majority class, this produced the balanced **Silver** dataset.

## Datasets

All files are CSVs with two columns: `sentence` and `label` (`1` = suggestion, `0` = non-suggestion).

| Dataset | Path | Size | Positive / Negative | Labeling |
|---|---|---|---|---|
| Initial | `datasets/initial_dataset/initial_dataset.csv` | 2,000 | 340 / 1,660 | Manual |
| Gold | `datasets/gold_dataset/gold_dataset.csv` | 2,400 | 1,200 / 1,200 | Manual + translated and reviewed |
| Silver | `datasets/parssugg_dataset/silver_dataset.csv` | 15,918 | 7,959 / 7,959 | Automatic (ParsBERT) |
| Silver (imbalanced) | `datasets/parssugg_dataset/silver_dataset_imbalance.csv` | 65,208 | 7,959 / 57,249 | Automatic (ParsBERT) |
| Unlabeled | `datasets/unlabeled_data/raw_data.csv` | ~65K | none | Raw sentences collected before labeling |

**ParsSugg** = Silver (training set) + Gold (test set). Both are in `datasets/parssugg_dataset/`.

## Results

F-score on the suggestion class (precision and recall are in the paper):

| Classifier | Initial | Gold | ParsSugg |
|---|---|---|---|
| SVM | 72.89 | 92.59 | 94.52 |
| Random Forest | 74.88 | 93.16 | 93.57 |
| CNN | 72.11 | 94.16 | 94.30 |
| LSTM | 77.37 | 94.91 | 93.77 |
| **ParsBERT** | **86.33** | **97.32** | **97.27** |

## Repository structure

```
├── classifiers/       Jupyter notebooks, one folder per model
│   ├── SVM/           svm_{initial,gold,parssugg}_dataset.ipynb
│   ├── RandomForest/  rf_{initial,gold,parssugg}_dataset.ipynb
│   ├── CNN/           cnn_{initial,gold,parssugg}_dataset.ipynb
│   ├── LSTM/          lstm_{initial,gold,parssugg}_dataset.ipynb
│   └── ParsBERT/      parsbert_*.ipynb (includes the notebook that generates the Silver dataset)
├── datasets/          Initial, Gold, ParsSugg (Silver + Gold) and unlabeled data
├── module/            preprocess.py: normalization, stop-word removal, stemming, emoji removal
└── other/             GridSearchCV results for SVM/RF on the Initial and Gold datasets
```

## Getting started

The notebooks were written for **Google Colab** and read data from Google Drive paths such as `/content/drive/MyDrive/data/gold_dataset.csv`. To run them:

1. Upload the CSV files you need (and `module/preprocess.py`) to your Drive, or edit the paths in the notebooks.
2. Install the dependencies (the notebooks do this in their first cells):

```bash
pip install transformers tensorflow scikit-learn pandas numpy matplotlib hazm demoji
pip install https://github.com/htaghizadeh/PersianStemmer-Python/archive/master.zip
```

3. Open a notebook from `classifiers/` and run it. ParsBERT is loaded from Hugging Face as `HooshvareLab/bert-base-parsbert-uncased`, and fine-tuning needs a GPU.

To evaluate your own model on ParsSugg, train on `silver_dataset.csv` and test on `gold_dataset.csv`.

## Citation

If you use the code or the datasets, please cite:

> Safari, L., Mohammady, Z. A semi-supervised method to generate a Persian dataset for suggestion classification. *Lang Resources & Evaluation* **58**, 839–858 (2024). https://doi.org/10.1007/s10579-023-09688-7

```bibtex
@article{safari2024parssugg,
  title   = {A semi-supervised method to generate a Persian dataset for suggestion classification},
  author  = {Safari, Leila and Mohammady, Zanyar},
  journal = {Language Resources and Evaluation},
  volume  = {58},
  pages   = {839--858},
  year    = {2024},
  doi     = {10.1007/s10579-023-09688-7}
}
```

## License

- **Code** (notebooks and `module/`): [MIT License](LICENSE)
- **Datasets** (`datasets/`): [CC BY 4.0](datasets/LICENSE). You may share and adapt them, including commercially, as long as you give credit by citing the paper above.

## Authors

Leila Safari and Zanyar Mohammady, Department of Computer Engineering, University of Zanjan, Iran.
Contact: lsafari@znu.ac.ir, zanyarmohammady@znu.ac.ir

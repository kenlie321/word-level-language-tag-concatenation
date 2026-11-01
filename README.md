# Word-Level Language Tag Placement for Indonesian-English Code-Switched Sentiment Analysis

Code and results for the paper *The Effect of Word-Level Language Tag Placement on Sentiment Analysis of Indonesian-English Code-Switched Text* (Smart Techno).

The study builds a word-level language identification (LID) module for Indonesian-English text and tests four ways of giving its tags to a sentiment classifier:

| Representation | Example input |
|---|---|
| Baseline | `interface friendly fitur lengkap` |
| Interleaved | `interface EN friendly EN fitur ID lengkap ID` |
| Adjacent | `interface friendly fitur lengkap [SEP] EN EN ID ID` |
| Concatenated | `interface_EN friendly_EN fitur_ID lengkap_ID` |

Each representation is used to fine-tune BERT (`bert-base-uncased`), mBERT (`bert-base-multilingual-cased`) and IndoBERT (`indolem/indobert-base-uncased`), and to train a Linear SVM with TF-IDF. Every configuration is run with five seeds (42, 123, 2024, 7, 99), giving 60 transformer runs and 20 SVM runs. Macro F1 is the confirmatory metric; paired t-tests with Holm correction compare configurations.

## Data and external resources

Three external resources are needed. None of the licensing for the second and third is stated by their authors, so they are linked rather than redistributed here.

| Resource | Used for | Source |
|---|---|---|
| Code Mixed Dataset (Indonesian-English), Tentua & Sprapto (2025), CC BY 4.0 | Sentiment classification and LID validation sample | https://data.mendeley.com/datasets/4w6ry5rfxs/1 |
| Indonesian-English code-mixed corpus of Barik et al. (2019) | Gold-standard validation of the LID module (Table 1) | https://github.com/seelenbrecher/code-mixed-normalization |
| KBBI v6.1.0 word list | Indonesian lexicon of the LID module | https://github.com/aryakdaniswara/kbbi-v6-wordlist |

How the input files are named and built (see also `data/README.md`):

1. **Dataset.** Download the Mendeley dataset and extract the `Sentiment Analysis Corpus` folder, which contains `train_new.csv`, `val_new.csv` and `test_new.csv` (columns `text`, `sentiment`). The three files are concatenated (19,767 rows) and deduplicated (19,547 rows) by the dedicated cell in the notebook, which writes `code_switched_en_id_combined_deduplicated.csv`.
2. **Filtering.** The preprocessing section removes meaningless statements (61 rows), statements of 100 words or longer (12 rows), and statements in which the LID module finds no English or no Indonesian token (4,460 rows). The result, 15,014 rows, is saved as `code_switched_en_id_preprocessed_filtered.csv` and is the input to all experiments.
3. **Gold standard.** Download `825_Indonesian_English_CodeMixed.csv` from the Barik et al. repository.
4. **KBBI.** Download `all_entries_v6.1.0.txt` from the KBBI repository and save it as `kbbi_word_list.txt`, the name the notebook reads.
5. The NLTK `words` corpus (236,736 entries) is downloaded by the notebook itself.

## Environment

All experiments ran on Google Colab with a Tesla T4 GPU (Python 3.13). The main library versions are PyTorch 2.11.0 (CUDA 13.0 build), transformers 5.21.0, tokenizers 0.23.3, accelerate 1.15.0, datasets 4.8.5, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, nltk 3.9.1 and lingua-language-detector 2.2.0. The complete list is in `requirements_colab.txt`, which the first cells of the notebook regenerate with `pip freeze`.

The first experiment run did not record library versions. The versions above are those of the later environment in which the results were verified (see Reproducibility).

## How to run

The notebook is organised in sections that should be run in this order, even though the training section appears first in the file:

1. **Environment and imports.** Records the library versions and installs `lingua-language-detector`.
2. **Data preparation.** Builds the combined, deduplicated file; runs the LID module over every statement; applies the three filters; saves the 15,014-row file.
3. **LID validation.** Evaluates the module on the Barik et al. gold standard (Table 1) and on the manual sample from the target dataset (Table 2). Sentences for the second were drawn in two rounds (50 and 100), stratified by sentiment class with equal allocation, excluding sentences already drawn.
4. **Token length statistics** (Table 5).
5. **SVM and transformer training.** Runs every configuration for the five seeds; results are appended to `multi_seed_results.csv`, and finished runs are skipped when the notebook is restarted.
6. **Statistical analysis.** Summary statistics, Shapiro-Wilk diagnostics, the paired t-tests and Holm correction (Tables 6-8).

Fine-tuning all 60 transformer runs takes many hours on a single T4. The results files in `results/` let the analysis in step 6 be reproduced without retraining.

Data split: random, not stratified, 70:10:20 with `random_state=42`, which gives a test set of 3,003 rows. Training uses a batch size of 32, 4 epochs, a learning rate of 2e-5, weight decay 0.01 and a maximum sequence length of 400. The checkpoint with the highest validation macro F1 is used for testing.

## Reproducibility

The Linear SVM results were reproduced exactly: all four representations and five seeds gave identical confusion matrices to the original run. The BERT baseline with seed 42 was reproduced to within one test prediction (macro F1 0.8902 against 0.8901), a difference expected from GPU fine-tuning across library versions. The verification runs are in `multi_seed_results.csv`.

The notebook also contains a grid search over SVM settings that was run after the reported results had been produced. It is analysis only; the reported configuration is the one stated in the paper (TF-IDF unigrams and bigrams, 20,000 features, `min_df=5`, `C=1`, balanced class weights, `max_iter=10000`).

## Citation

If you use this code, please cite the paper (full reference to be added on publication) and the resources it builds on:

- Tentua, M. N., & Sprapto, S. (2025). *Code Mixed Dataset (Indonesian-English)* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/4w6ry5rfxs.1
- Barik, A. M., Mahendra, R., & Adriani, M. (2019). Normalization of Indonesian-English code-mixed Twitter data. *Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)*, 417-424.
- KBBI v6.1.0 word list. https://github.com/aryakdaniswara/kbbi-v6-wordlist

## License

The code is released under the MIT license (see `LICENSE`). The Mendeley dataset is distributed under CC BY 4.0. The Barik et al. corpus and the KBBI word list are not redistributed here and remain under their authors' terms.

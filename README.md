# DSA4213 Assignment 1: Reproduction Guide

This repository contains the notebooks and saved model artifacts for a language-modeling study on WikiText-2, followed by a Yelp Polarity sentiment-transfer experiment. The notebooks are designed for Google Colab. Run their cells from top to bottom in the order below.

## Contents

| Path | Purpose |
| --- | --- |
| `notebooks/01_data_preparation.ipynb` | Download and prepare the shared WikiText-2 corpus |
| `notebooks/02_ngram_30_config_colab.ipynb` | Fit and evaluate the interpolated n-gram model |
| `notebooks/03_lstm_colab_tuned.ipynb` | Tune, train, and evaluate the Part I LSTM |
| `notebooks/04_rnn_colab_tuned.ipynb` | Tune, train, and evaluate the Part I RNN |
| `notebooks/05_transformer_colab_tuned.ipynb` | Tune, train, and evaluate the Part I Transformer |
| `notebooks/06_part2_lstm_random_baseline.ipynb` | Reuse the selected LSTM with random trainable embeddings |
| `notebooks/07_part2_lstm_cbow_frozen.ipynb` | Evaluate the LSTM with frozen CBOW embeddings |
| `notebooks/08_part2_lstm_glove_frozen.ipynb` | Evaluate the LSTM with frozen GloVe embeddings |
| `notebooks/09_part3_restaurant_sentiment.ipynb` | Compare the frozen LSTM representations on Yelp Polarity |
| `configs/common.json` | Shared data and tokenization configuration |
| `runs/` | Saved model checkpoints, selected configurations, and run metadata |
| `embeddings/` | Prepared embedding artifacts used by Part II |
| `partIII/yelp_polarity/` | Part III caches and recorded experiment outputs |

The Part III task uses the mixed-business **Yelp Polarity** dataset, not a restaurant-only dataset.

## Run environment

The notebooks are intended to run on a Google Colab runtime. A GPU runtime (such as a T4, when available) is recommended for training the neural language models; CPU execution may take considerably longer. Part III also works with a CPU runtime, though feature extraction will be slower.

The notebooks install or use the packages they need. The principal dependencies are:

- Python and Jupyter (provided by Colab)
- PyTorch
- `datasets`, `huggingface_hub`
- `numpy`, `pandas`, `pyarrow`
- `matplotlib`, `seaborn`
- `psutil`, `threadpoolctl`, `tqdm`
- `gensim` for CBOW embedding preparation
- `scikit-learn` and `joblib` for the Part III classifier

Notebook 01 installs `datasets`, `huggingface_hub`, `numpy`, and `pandas`; notebook 02 installs the n-gram dependencies; notebook 09 installs its dataset and classifier dependencies. Colab supplies PyTorch. The run folders include environment and lock files documenting package versions from the recorded executions; these are reference environments, and hosted Colab package versions and hardware can change.

## Option A: Run in Google Colab in a browser

1. Upload this repository to Google Drive, or copy it into a Drive folder named `DSA4213_Assignment1`.
2. Open the notebooks from Drive in Colab. Alternatively, upload a notebook to Colab and make the repository files available in Drive.
3. Run the notebook's dependency-installation cell, if it has one. If a package update prompts for a runtime restart, restart and continue from the first cell.
4. In the Drive setup cell, mount Google Drive and set `PROJECT_ROOT` to the folder containing this repository. The notebooks currently use:

   ```python
   PROJECT_ROOT = Path('/content/drive/MyDrive/DSA4213_Assignment1')
   ```

   If the repository is stored elsewhere, change this path in each notebook that defines `PROJECT_ROOT`.
5. Run all cells from top to bottom. Keep the same `PROJECT_ROOT` throughout so the notebooks can reuse the prepared data and saved runs.

## Option B: Run Colab notebooks from Visual Studio Code

The Google Colab extension connects VS Code to a hosted Colab runtime. It is different from running the notebook on your computer; notebook code executes on the selected Colab server.

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. In VS Code, open **Extensions** (`Ctrl+Shift+X` on Windows/Linux), search for **Jupyter** by Microsoft and install it.
3. Search for **Google Colab** by Google and install the [Google Colab VS Code extension](https://marketplace.visualstudio.com/items?itemName=google.colab).
4. Open this repository as a VS Code folder and open a notebook under `notebooks/`.
5. When prompted, sign in to your Google account. In the notebook's kernel picker, choose **Select Kernel > Colab > Auto Connect** and authorize the connection.
6. Make the repository files available to the Colab server. Upload/copy the project folder to Google Drive, then set `PROJECT_ROOT` in the notebook to its Drive location. The extension's **Colab: Mount Google Drive to Server...** command can insert the Drive-mount cell; alternatively, use the notebook's existing Drive-mount setup cell.
7. Run the notebook cells in order. The first dependency cell may update runtime packages; restart the runtime if prompted.

If the Colab kernel option is not shown, ensure the Google Colab and Jupyter extensions are enabled, sign in, reopen the notebook, and select the kernel again. See the [extension quick start](https://marketplace.visualstudio.com/items?itemName=google.colab) for current UI details.

## Data access and preprocessing

### WikiText-2 (Parts I and II)

Notebook 01 loads `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, using the Hugging Face `datasets` library. Internet access is needed the first time the dataset is not already cached. The prepared corpus and vocabulary are written below `data/` under the selected project root and reused by the later notebooks.

The shared settings are recorded in `configs/common.json`. In brief:

- Text is handled as nonempty original rows, including headings.
- Normalization applies Unicode NFKC, converts curly apostrophes to straight apostrophes, lowercases, and collapses whitespace.
- The tokenizer is the repository's Unicode regular-expression tokenizer (`unicode_regex_v1`).
- Exact normalized duplicate rows are removed with test-before-validation-before-train precedence.
- The vocabulary is capped at 10,000 tokens with minimum frequency 2. Special IDs are `PAD=0`, `UNK=1`, `BOS=2`, and `EOS=3`.
- Model blocks have length 64; recurrent state is reset at block boundaries.

Notebook 01 records data manifests and checksums so later runs can verify the prepared inputs. Do not edit the vocabulary or processed files between model runs if comparing the reported experiments.

### Yelp Polarity (Part III)

Notebook 09 downloads `fancyzhx/yelp_polarity` from the Hugging Face Datasets Hub when no compatible dataset cache exists. The recorded run used revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`. For an exact data-source reproduction, set `DATASET_REVISION` in notebook 09's controls to this commit SHA; its default (`None`) resolves the dataset revision available at the time of the new run. Internet access is needed for a fresh run. The official train and test partitions are preserved; the sampled experiment uses 8,000 training, 1,000 validation, and 1,000 test reviews, with seed 42. Rating-derived labels are negative or positive; neutral reviews are omitted.

The notebook limits reviews to 256 lexical tokens and extracts frozen top-layer LSTM representations in non-overlapping blocks of length 64. Mean pooling excludes PAD, BOS, and EOS, and retains UNK. It fits a training-only `StandardScaler` plus L2 logistic regression and selects among `C = [0.01, 0.1, 1.0, 10.0]` by validation macro-F1, then validation accuracy, then smaller `C`. Test results are for final evaluation, not for tuning.

## Reproduction order and commands

In Colab, open each notebook and choose **Runtime > Run all** (or use the VS Code notebook's **Run All** action). Run these notebooks in order:

1. `01_data_preparation.ipynb`
2. `02_ngram_30_config_colab.ipynb`
3. `03_lstm_colab_tuned.ipynb`
4. `04_rnn_colab_tuned.ipynb`
5. `05_transformer_colab_tuned.ipynb`
6. `06_part2_lstm_random_baseline.ipynb`
7. `07_part2_lstm_cbow_frozen.ipynb`
8. `08_part2_lstm_glove_frozen.ipynb`
9. `09_part3_restaurant_sentiment.ipynb`

Parts II and III depend on files produced by earlier notebooks. In particular, Part III requires the three Part II LSTM run folders with their checkpoints and supporting model/tokenizer files. Keep the `runs/` and `embeddings/` folders in the project root when reusing the recorded checkpoints.

### Reusing checkpoints versus retraining

- **Reproduce the saved neural-model evaluations:** Keep the appropriate run folders and data artifacts in place. In Part I notebooks 03-05, set `TRAIN = False` to skip fitting and evaluate/generate from the saved run; the expected `RUN_ID` and saved files must exist under `PROJECT_ROOT`. Notebook 02 is the separate n-gram workflow and does not use this neural-model `TRAIN` control.
- **Train the Part I neural models again:** In notebooks 03-05, use `TRAIN = True` and `MODE = 'auto'`. The notebooks can resume/reuse compatible completed runs. Use a new `RUN_ID` for a deliberately new experiment rather than overwriting an existing run. Training and tuning can take a long time.
- **Run Part III for the first time:** Leave `EXTRACTION_MODE = 'auto'` and `CLASSIFIER_MODE = 'auto'` in notebook 09. It downloads/builds missing data, extracts features from the Part II checkpoints, and fits the classifier. Internet access is required if the dataset is not cached.
- **Reuse Part III caches:** Set both modes to `reuse_only` only when all compatible dataset, feature, and classifier caches already exist. This mode does not create missing caches.

Changing run IDs, seeds, data, model settings, or caches creates a different experiment and may not reproduce the recorded results.

## Main experiment settings and random seeds

The shared data seed is 42. The Part I neural-model settings are saved with each run; the selected configurations are:

| Model | Selected configuration |
| --- | --- |
| LSTM | 100-d embeddings, hidden size 256, 1 layer, dropout 0.2, batch size 32, learning rate 0.001 |
| RNN | 100-d embeddings, hidden size 128, 1 layer, dropout 0.2, batch size 32, learning rate 0.003 |
| Transformer | 100-d embeddings, 2 layers, feedforward size 800, dropout 0.1, batch size 32, learning rate 0.003 |

All three neural models use up to 50 epochs, Adam, gradient clipping at norm 1, seed 42, shuffle seed 43, generation seed 123, validation-based checkpoint selection, early-stopping patience 7, and a plateau scheduler with factor 0.5 and patience 2. The seven-trial staged search varies learning rate, dropout, and model width, as described in the notebooks.

The n-gram model uses seed 42 and selects smoothing/interpolation settings on validation cross-entropy; its candidate grids are in `runs/I_ngram_05cfd29bc588/config.json`.

Part II reuses the selected Part I LSTM settings (seed 42) for random-trainable, CBOW-frozen, and GloVe-frozen embedding variants. Part III uses seed 42 for its Yelp sample and classifier. Its candidate-selection and pooling settings are described above and recorded in `partIII/yelp_polarity/runs/75004512bd2b2c41a798/configuration.json`.

The experiments use one seed for the reported comparisons. Different Colab hardware, library versions, or nondeterministic GPU operations may cause small numerical differences.

## Outputs

The notebooks save checkpoints, metrics, run configurations, and manifests under `runs/`, `embeddings/`, `data/`, and (for Part III) `partIII/yelp_polarity/`, relative to `PROJECT_ROOT`. Part I evaluation outputs are also consolidated under `results/` when the applicable notebooks have run. Rerunning a notebook regenerates outputs that are missing, provided its inputs and dependencies are available.

The final assignment report is in `output/report/DSA4213_Assignment1_Report.docx` in the working copy. Intermediate report-building files are not required to run the notebooks.

## Troubleshooting

- **Notebook cannot find a file:** Confirm that `PROJECT_ROOT` points to the project folder, not to its parent or to `notebooks/`.
- **Part II/III reports missing checkpoints:** Run the prerequisite notebooks or restore the corresponding saved folders from `runs/` and `embeddings/`.
- **Hugging Face download fails:** Confirm that the Colab runtime has internet access and rerun the cell. Do not use `reuse_only` until compatible caches exist.
- **Package update requests a restart:** Restart the runtime and rerun the notebook from its first cell.
- **Runtime disconnects during training:** Keep `PROJECT_ROOT` on mounted Drive so saved checkpoints and progress files persist.

## References

- WikiText-2: [Hugging Face Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- Yelp Polarity: [Hugging Face dataset card](https://huggingface.co/datasets/fancyzhx/yelp_polarity)
- Colab in VS Code: [Google Colab VS Code extension](https://marketplace.visualstudio.com/items?itemName=google.colab)
- Colab use: [Google Colaboratory](https://colab.research.google.com/)

# DSA4213 Part II: random_trainable
This run uses the selected Part I configuration and unchanged prepared data.
Project root: /content/drive/MyDrive/DSA4213_Assignment1
Baseline study: I_lstm_tuned_s42_v2; selected trial: trial_07_hidden_size.
Best epoch: 47; maximum epochs: 50.

## Run / reload
Run 01_data_preparation and complete the tuned Part I LSTM first. Keep their Drive outputs.
Upload the Part II notebook to Colab, select T4 GPU for new LSTM training, check PROJECT_ROOT, Run all.
TRAIN=True, MODE='auto' resumes incomplete LM training from the last completed epoch and reuses completed training.
TRAIN=False performs no fitting; it requires completed artifacts for frozen variants.
Baseline notebook loads Part I weights without fitting. Existing Part I artifacts are not changed.
If CBOW preparation is interrupted before COMPLETE.json, rerunning rebuilds its ten-epoch training from scratch.
Do not manually change saved configs/checksums. Use new run/embedding IDs for deliberate new experiments.

## Reusable inference / Part III encoder
Keep best.pt with lstm_model.py, vocab.json and preprocessing.py.
```python
import torch
from lstm_model import LSTMLanguageModel
package = torch.load('best.pt', map_location='cpu', weights_only=True)
model = LSTMLanguageModel(package['config'])
model.load_state_dict(package['model_state'])
model.eval()
# IDs: use the saved tokenizer/vocabulary, BOS=2, PAD=0; right padding only.
# with torch.inference_mode(): logits = model(input_ids)
# features = model.encode_mean(input_ids) excludes PAD/BOS/EOS; UNK remains.
```
The full frozen embedding table is included in best.pt, so inference needs no public-vector download.

## Outputs and measurement definitions
config.json: immutable settings and baseline/embedding hashes.
best.pt: selected inference package. weights.pt: pure state dict. last.pt: frozen-run recovery checkpoint.
history.csv, learning_curves.png, stability.png: training, validation, learning-rate and gradient evidence.
initialization_audit.json: same non-embedding initialization; frozen table unchanged.
lm_metrics.csv: token-weighted test CE in nats and exp(CE), with PAD excluded and EOS/UNK scored.
generation.jsonl: same saved prompts and decoding as Part I (greedy and temperature .8/top-50; at most 50 tokens).
benchmark*.csv: same fixed CPU/GPU workload; compare only compatible device rows.
coverage*.csv: fixed-vector coverage; ordinary vocabulary excludes special IDs.
embedding_record.json: separate own-embedding training cost; public pretraining cost is external/unmeasured.
sessions/: hardware, versions, checkpoint timing, and per-execution evidence.
results/part_II_lstm/: automatically consolidated results when all three notebooks have run.
LM training time includes validation; setup/download and checkpoint I/O are separate.
Baseline cost is the original selected trial cost; its seven-trial search is recorded separately in baseline provenance.

## Interpretation
Single seed; gradient curves characterize this run, not seed-to-seed robustness.
Configuration was selected using the random baseline. It may favor that setting.
CBOW versus GloVe also changes training objective, corpus size/domain and vector scale.
Random-trained versus frozen changes both initialization and trainability; do not attribute all effects to pretraining.
Public embeddings may have corpus overlap that cannot be audited from released vectors.
Validation selects LM epochs; test is for final reporting, never tuning.

## Attribution
User's tuned Part I notebook and data-preparation notebook supply the architecture and data contract.
Assignment1_2026 specifies the comparison. Substantial coding assistance: ChatGPT.
Gensim Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html
GloVe (Pennington et al., 2014): https://nlp.stanford.edu/projects/glove/
Gensim-data packaging: https://github.com/piskvorky/gensim-data
PyTorch: https://docs.pytorch.org/docs/stable/index.html
Write your own report interpretation from the measured outputs.

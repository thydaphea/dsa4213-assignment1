# trial_07_dim_feedforward: DSA4213 Part I Transformer
Data version: beacb3de63dc4261
Best epoch: 33; completed epochs: 40/50

## Run in Colab
1. Run 01_data_preparation.ipynb and retain its Drive outputs.
2. Open 05_transformer_colab_tuned.ipynb, select a GPU runtime and set PROJECT_ROOT.
3. First run: TRAIN=True, MODE='auto'; Run all.
4. Later: TRAIN=False; Run all. No training occurs, including the diagnostic.
5. Recovery: TRAIN=True, MODE='auto' resumes from last completed epoch.
   A completed run is loaded automatically; choose a new RUN_ID for a new experiment.
6. Save this notebook into your Drive notebooks folder too.

## Dependencies and data access
PyTorch, NumPy, pandas, Matplotlib, tqdm; exact installed versions in requirements-lock.txt.
Use a device-compatible PyTorch installation; a Colab freeze is an environment record,
not a guarantee that a CUDA build installs unchanged on another platform.
Corpus: Salesforce/wikitext, wikitext-2-raw-v1; access/revision in source.json.
Data preparation: normalization, Unicode regex, train-only vocabulary, deduplicated
original splits, independent shifted blocks. Full settings: common_data_config.json.
No raw dataset is downloaded or reprocessed by this notebook.
Main/init seed 42; shuffle seed 43; sampling seed 123.

## Load without training (from this folder)
```python
import torch
from transformer_model import CausalTransformerLM
pkg = torch.load('best.pt', map_location='cpu', weights_only=True)
model = CausalTransformerLM(pkg['config'])
model.load_state_dict(pkg['model_state'])
model.eval()
# Use preprocessing.py tokenize() and vocab.json token_to_id for new text.
# Inputs are int64 IDs shaped [batch, length]. Prepend BOS=2 and right-pad with PAD=0.
# For next-token generation use at most 64 most-recent context tokens; restart positions on each call.
# logits = model(input_ids) with torch.inference_mode().
# For Part III mean features: model.encode_mean(input_ids); exclude BOS/PAD/EOS.
```

## Artifacts
best.pt: inference weights + architecture and provenance.
weights.pt: pure state dictionary used for inference-size measurement.
last.pt: current model, optimizer, history, RNG/shuffle states, best weights.
initial_weights.pt: original untrained embedding and non-embedding weights for Part II.
transformer_model.py: reusable model/encoder source; preprocessing.py and vocab.json: input contract.
history.csv and learning_curves.png: convergence/stability evidence.
lm_metrics.csv, generation.jsonl, benchmark*.csv: results when enabled.
sessions/: hardware, environment, timing and result records from each execution.
artifact_manifest.json: inventory with file hashes (does not hash itself).

## Measurement definitions and limitations
CE = summed NLL / scored targets; PPL = exp(CE); score EOS/UNK, exclude PAD.
Output normalization excludes PAD/BOS. Generation also suppresses UNK.
Training time = train loop + in-loop validation, excluding checkpoint I/O/setup.
Timing summary records measured completed-epoch work; unsaved interrupted work is absent.
Inference uses saved 4 prompts, 50 tokens each, greedy, no EOS stop,
one warm-up + three repeats, no key/value cache. CPU/GPU rows are separate.
Selection: lowest validation CE; earlier epoch wins ties; no test-based tuning.
Staged validation tuning, not parameter-matched. Single-seed result.
Total search compute seconds: 10058.44291657001.
Selection record: /content/drive/MyDrive/DSA4213_Assignment1/runs/I_transformer_tuned_s42_v2/selection.json.
Stopping reason: early_stopping. Context resets every block.
Part II reuse requires the exact baseline configuration and data. Train frozen-embedding
variants' non-embedding weights from scratch or from initial_weights.pt, never best.pt.
Part III feature extraction uses frozen encoder states; classifier experiments are separate.

## Attribution
Assignment1_2026(1).pdf and DSA4213_Assignment1_Implementation_Plan.pdf supplied the requirements/design.
01_data_preparation.ipynb supplied the actual saved-data schema, tokenizer and four prompts.
PyTorch Transformer layer API: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
PyTorch checkpoint tutorial: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
WikiText: Merity et al. (2016), Pointer Sentinel Mixture Models.
This notebook was prepared with substantial ChatGPT coding assistance. Review the code,
acknowledge assistance, and write your own analysis of the measured results.

# Reload I_ngram_05cfd29bc588 without training

Use the same Python/NumPy versions recorded in the original session where possible.
Install numpy pandas psutil threadpoolctl; mount Drive and keep the complete run folder.
Load only your own trusted pickle files. Verify inference.pkl against inference.sha256.json.

```python
import sys, json, pickle, hashlib
from pathlib import Path
run_dir = Path('/content/drive/MyDrive/DSA4213_Assignment1/runs/I_ngram_05cfd29bc588')
sys.path.insert(0, str(run_dir))
import ngram_core_b7cd39e9e759 as core
path = run_dir / 'inference.pkl'
expected = json.loads((run_dir / 'inference.sha256.json').read_text())['sha256']
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
with path.open('rb') as f:
    package = pickle.load(f)
model = package['model']
# Pass saved x/y blocks for scoring. Do not rebuild their vocabulary.
# counts, totals = model.counts_for_targets(x, y)
# metrics = model.score_counts(counts, totals)
# output = core.generate(model, prompt_ids, decoding='greedy')
```

For exact evaluation provenance reuse common.json, data_complete.json and the
original processed files; verify their checksums and scored target hash.
Use inference_workload.json for the same generation benchmark in notebook 03/05.
N-gram counts use one-token history at block starts and at most two elsewhere.
Count fitting is one training pass; no neural parameter count or learning curve exists.
The four saved prompts are preserved from notebook 01 (200 benchmark tokens/repeat).

Training timing and its machine live in training_cost.json; current session
hardware lives in sessions/<session>/environment.json. Never relabel load time as
training time. Compare CPU inference on one shared session; GPU results separately.

Sources: supplied Assignment1_2026(1).pdf (Part I, reproducibility/tool use),
DSA4213_Assignment1_Implementation_Plan.pdf (Sections 6, 8–11, 14–15), and
01_data_preparation.ipynb (file/schema/tokenizer contract). Dataset provenance
is preserved in data_source.json. This implementation was prepared with
ChatGPT assistance; review and acknowledge substantial assistance in submission.

# Part III: Yelp sentiment from frozen LSTM representations

Run 09_part3_restaurant_sentiment.ipynb in Colab from top to bottom after Part II 06/07/08.
The task uses mixed-business Yelp Polarity, not restaurant-only data.

1. Run dependency cell; restart if Colab requests it.
2. Mount Drive and set PROJECT_ROOT / RUN_DIRS in Section 2.
3. First run: EXTRACTION_MODE=auto and CLASSIFIER_MODE=auto.
4. Later: reuse_only for both switches loads only compatible caches.
5. Do not change settings after inspecting test results to optimize test scores.

Inputs: each Part II run needs best.pt, config.json, training_summary.json,
lstm_model.py, vocab.json, preprocessing.py, common_data_config.json.
Optional metrics.json supplies next-token CE/PPL; vocabulary_coverage.csv supplies embedding-source coverage.
No CBOW training file, GloVe download, or original raw WikiText files are needed here.

Source: https://huggingface.co/datasets/fancyzhx/yelp_polarity
Pinned revision: bbf1c97a1f0cf005e5aded43839fd814654a1557
Labels: 0 negative, 1 positive; official train/test preserved.
Selected counts: {'train': 8000, 'validation': 1000, 'test': 1000}; seed: 42.
Data cache: /content/drive/MyDrive/DSA4213_Assignment1/partIII/yelp_polarity/datasets/07f2cabd6c44f2f7e813
Experiment outputs: /content/drive/MyDrive/DSA4213_Assignment1/partIII/yelp_polarity/runs/75004512bd2b2c41a798
LM block length: 64; retained review limit: 256 lexical tokens.
Mean pooling: top LSTM layer; PAD/BOS/EOS excluded; UNK included; token-weighted across chunks.
All LMs frozen, eval mode, inference_mode; no LM optimization.
Classifier: training-fitted StandardScaler + L2 logistic regression.
C grid: [0.01, 0.1, 1.0, 10.0]; validation macro-F1 selection, accuracy then smaller C tie-break.
Selected candidate remains training-only fitted; threshold 0.5; no train+validation refit.

Use configuration.json, model_artifacts.json, dataset_source.json and split_manifest.csv to audit inputs.
Inspect sessions/*/environment.json and requirements-lock.txt for hardware and dependency versions.
Cached metadata retains original compute times; a cache hit is not reported as zero training cost.
Cache keys include checkpoint, split content/order/labels, tokenizer/vocabulary, and method configuration.
Run output manifest references the compatible cached feature and classifier files.

Limitations: one LM seed and one labeled split; mixed businesses; no business/user grouping;
WikiText-to-Yelp domain shift; truncated reviews; chunk boundaries; mean-pooling information loss;
rating-derived polarity; neutral reviews omitted; possible near-duplicates; pretrained GloVe exposure not audited.
Input settings combine embedding source and trainability, not a pure source-only causal ablation.
Three ranking points do not support a strong correlation or statistical-generalization claim.

Sources: Zhang, Zhao & LeCun (2015), Character-level Convolutional Networks for Text Classification.
Hugging Face dataset card and loading docs: https://huggingface.co/docs/datasets/loading
Logistic regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
Adapted from the user's Part I/II LSTM and preprocessing notebooks, with coding-assistant help.
Read and explain the implementation and write interpretation in your own words.

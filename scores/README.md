# Results

Here we present results of our baselines and advanced models.

## CoNLL
### Baseline
Date: 02.08.2021
What: Base BERT with only Linear layer for baseline results on pure CoNLL Dateset.

<details>
  <summary>Model and Hyperparameters</summary>

    'model': 'Bert-Base-Cased',
    'corpus': 'CoNLL2003',
    'document_context': False,
    'hidden_size': 768,
    'batch_size': 32,
    'shuffle_train': True,
    'shuffle_eval': True,
    'shuffle_test': False,
    'optimizer': 'AdamW',
    'learning_rate': 1e-5,
    'epochs': 5,
    'last_epoch_lstm': False,
    'seed': [693, 42, 3021, 218, 1421]

</details>

Metrics

|Metric  |Score|STD |
|--------|-----|----|  
|Token F1|94.30|0.12|
|Span  F1|89.85|0.22|
|ORG     |86.64|0.25|
|PER     |95.89|0.27|
|LOC     |92.23|0.10|
|MISC    |78.59|0.79|
import nltk

nltk.download('punkt')
import numpy as np

params = {
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
}

token_f1 = [94.43, 94.41, 94.17, 94.15, 94.36]
span_f1 = [90.09, 89.84, 89.71, 89.53, 90.09]
loc_f1 = [92.41, 92.15, 92.28, 92.13, 92.19]
per_f1 = [95.58, 95.76, 96.20, 95.70, 96.22]
org_f1 = [86.99, 86.62, 86.41, 86.33, 86.87]
misc_f1 = [79.45, 79.06, 77.81, 77.46, 79.15]

print('-' * 20)
print('Experiments Info:')
print('-' * 20)
for param in params:
    print(param, params[param])

print('-' * 20)
print('Results:')
print('-' * 20)
print('Mean Token F1 Score {:.2f}, STD {:.2f}'.format(np.mean(token_f1), np.std(token_f1)))
print('Mean Span  F1 Score {:.2f}, STD {:.2f}'.format(np.mean(span_f1), np.std(span_f1)))
print('Mean LOC   F1 Score {:.2f}, STD {:.2f}'.format(np.mean(loc_f1), np.std(loc_f1)))
print('Mean PER   F1 Score {:.2f}, STD {:.2f}'.format(np.mean(per_f1), np.std(per_f1)))
print('Mean ORG   F1 Score {:.2f}, STD {:.2f}'.format(np.mean(org_f1), np.std(org_f1)))
print('Mean MISC  F1 Score {:.2f}, STD {:.2f}'.format(np.mean(misc_f1), np.std(misc_f1)))
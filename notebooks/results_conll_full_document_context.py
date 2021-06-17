import numpy as np

print('Experiment Parameters')
print('Model: BERT Base + BiLSTM + Dropout(0.1) + Linear')
print('Batch Type: Sentence Collection + Full Document Context')
print('Shuffle: [Train, Dev, Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([93.44, 93.43, 93.19, 93.39, 93.18])
accuracy = np.array([94.41, 94.87, 94.71, 94.66, 95.00])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT Large + BiLSTM + Dropout(0.1) + Linear')
print('Batch Type: Sentence Collection + Full Document Context')
print('Shuffle: [Train, Dev, Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear

# [Final] Test : Loss = 0.13067, F1-score = 93.90%, Repeated Entities Accuracy = 95.34%: 100%|██████████| 108/108 [00:49<00:00,  2.18it/s]
# [Final] Test : Loss = 0.12706, F1-score = 93.85%, Repeated Entities Accuracy = 95.21%: 100%|██████████| 108/108 [00:48<00:00,  2.21it/s]
# [Final] Test : Loss = 0.12869, F1-score = 93.83%, Repeated Entities Accuracy = 95.46%: 100%|██████████| 108/108 [00:49<00:00,  2.19it/s]
# [Final] Test : Loss = 0.12850, F1-score = 93.74%, Repeated Entities Accuracy = 94.50%: 100%|██████████| 108/108 [00:49<00:00,  2.19it/s]
# [Final] Test : Loss = 0.12344, F1-score = 94.18%, Repeated Entities Accuracy = 94.75%: 100%|██████████| 108/108 [00:49<00:00,  2.20it/s]

f1_score = np.array([93.90, 93.85, 93.83, 93.74, 94.18])
accuracy = np.array([95.34, 95.21, 95.46, 94.50, 94.75])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)


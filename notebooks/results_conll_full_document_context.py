import numpy as np

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: YES,', 'Batch Type: Sentence Collection + Full Document Context, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([92.69, 91.71, 93.20, 92.65])
accuracy = np.array([93.74, 93.95, 94.20, 93.66])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: YES,', 'Batch Type: Sentence Collection + Full Document Context, Shuffle: [Train, Dev, Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([92.24, 91.46, 91.42, 92.35])
accuracy = np.array([93.53, 90.80, 91.18, 92.23])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)
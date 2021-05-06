import numpy as np

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features:  NO,', 'Batch Type: Sentence Collection, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([86.82, 87.33, 86.64])
accuracy = np.array([82.58, 84.11, 82.91])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: YES,', 'Batch Type: Sentence Collection, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([87.11, 87.04, 86.88])
accuracy = np.array([85.21, 83.76, 83.19])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

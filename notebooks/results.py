import numpy as np

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features:  NO,', 'Batch Type: Sentence Collection, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([92.51, 92.57, 91.65, 93.11, 92.67])
accuracy = np.array([92.69, 92.35, 89.41, 92.14, 92.10])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: YES,', 'Batch Type: Sentence Collection, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([92.05, 92.11, 92.35, 92.41, 92.66])
accuracy = np.array([92.10, 92.39, 93.03, 92.23, 93.19])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: YES,', 'Batch Type: Document, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([91.98, 91.56, 92.42, 92.13, 90.93])
accuracy = np.array([93.03, 87.75, 91.57, 90.91, 91.82])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: NO,', 'Batch Type: Document, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([91.21, 91.12, 92.03, 90.37, 92.51])
accuracy = np.array([91.53, 87.55, 91.74, 85.31, 92.28])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))
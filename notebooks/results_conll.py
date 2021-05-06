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
f1_score = np.array([91.54, 92.37, 90.22, 91.10, 91.40])
accuracy = np.array([92.40, 92.78, 90.00, 89.46, 92.90])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))

print('-' * 115)

print('Experiment Parameters')
print('Model: BERT + BiLSTM + Dropout(0.1) + Linear')
print('Context Features: NO,', 'Batch Type: Document, Shuffle: [Train, Dev]: True, [Test]: False, Batch Size = 32')
# BERT + BiLSTM + Dropout(0.1) + Linear
f1_score = np.array([90.74, 91.81, 91.20, 91.52, 89.47])
accuracy = np.array([91.41, 92.94, 91.53, 94.02, 89.62])

print('Mean F1-Score: {:.2f}, F1-Score std: {:.2f}'.format(f1_score.mean(), f1_score.std()))
print('Mean Accuracy: {:.2f}, Accuracy std: {:.2f}'.format(accuracy.mean(), accuracy.std()))
from typing import Dict

from seqeval.metrics import accuracy_score

class FMeasureStorage():
    def __init__(self, true_positive: int = 0,
                 false_positive: int = 0,
                 true_negative: int = 0,
                 false_negative: int = 0):

        self.true_positive = true_positive
        self.false_positive = false_positive
        self.true_negative = true_negative
        self.false_negative = false_negative

    def __iadd__(self, iteration_result: Dict[str, int]):
        self.true_positive += iteration_result['TP']
        self.false_positive += iteration_result['FP']
        self.true_negative += iteration_result['TN']
        self.false_negative += iteration_result['FN']

        return FMeasureStorage(self.true_positive,
                               self.false_positive,
                               self.true_negative,
                               self.false_negative)

    def print_rates(self):
        print('True Positives {} | False Positives {} | True Negatives {} | False Negatives {}'.format(
            self.true_positive, self.false_positive, self.true_negative, self.false_negative), end='\n')

    def report(self):
        precision = self.true_positive / max(1, (self.true_positive + self.false_positive))
        recall = self.true_positive / max(1, (self.true_positive + self.false_negative))
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall


class AccuracyStorage():
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def __add__(self, labels: dict):
        self.true_labels.extend(labels['true'])
        self.pred_labels.extend(labels['pred'])

    def report(self):
        return accuracy_score(self.true_labels, self.pred_labels)

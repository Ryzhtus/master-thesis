class MetricsStorage():
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def __add__(self, iteration_result: dict):
        self.true_positive += iteration_result['TP']
        self.false_positive += iteration_result['FP']
        self.true_negative += iteration_result['TN']
        self.false_negative += iteration_result['FN']

    def print_rates(self):
        print('True Positives {} | False Positives {} | True Negatives {} | False Negatives {}'.format(
            self.true_positive, self.false_positive, self.true_negative, self.false_negative), end='\n')

    def report(self):
        precision = self.true_positive / max(1, (self.true_positive + self.false_positive))
        recall = self.true_positive / max(1, (self.true_positive + self.false_negative))
        f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score, precision, recall
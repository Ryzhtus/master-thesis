from sklearn.metrics import f1_score

def calculate_score(predict_tags, correct_tags):
    predicted_labels = list(predict_tags.cpu().numpy())
    correct_labels = list(correct_tags.cpu().numpy())

    predicted_labels_without_mask = []
    correct_labels_without_mask = []
    for p, c in zip(predicted_labels, correct_labels):
        if c > 1:
            predicted_labels_without_mask.append(p)
            correct_labels_without_mask.append(c)

    return f1_score(correct_labels_without_mask, predicted_labels_without_mask, average="micro")
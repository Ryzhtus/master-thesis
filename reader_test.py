from collections import Counter
from ner_lightning.reader import ReaderCoNLL

reader = ReaderCoNLL()
documents = reader.parse("data/conll2003/IOB2/train.txt")

for document in documents:
    for sentence in document.sentences:
        for word in sentence:
            if word.feature_vector()[-1] == True:
                print(word, word.features, word.feature_vector())

"""
repeated_entities = Counter()
for document in documents:
    for sentence in document.sentences:
        for span in sentence.spans:
            if span.label != 'O':
                if span.tokens[0][0].islower():
                    repeated_entities[span.span] += 1


for span in sorted(repeated_entities, key=lambda x: repeated_entities[x], reverse=True):
    print(span, repeated_entities[span])"""
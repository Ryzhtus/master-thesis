from ner.reader import ReaderCoNLL, ReaderOntonotes
from collections import Counter

reader = ReaderOntonotes(include_document_ids=True)
docs, _, _, document_to_sentences, _ = reader.read('data/ontonotes/train')

tokenCounter = Counter()

for document_id in document_to_sentences.keys():
    for sentence_id in document_to_sentences[document_id]:
        tokenCounter[document_id] += len(docs[sentence_id])

doc_greater_512 = []
for value in tokenCounter.values():
    if value >= 512:
        doc_greater_512.append(value)

print('N Docs All = {}'.format(len(list(tokenCounter.keys()))))
print('N Docs > 512 = {}'.format(len(doc_greater_512)))
for item in doc_greater_512:
    print(item)
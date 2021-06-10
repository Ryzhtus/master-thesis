from named_entity_recognition.document import Document
from named_entity_recognition.reader import ReaderDocumentCoNLL
from named_entity_recognition.dataset import SentencesPlusDocumentsDataset

import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
reader = ReaderDocumentCoNLL()
sentences, tags, masks, document2sentences, sentence2position_in_document = reader.get_sentences('data/conll2003/train.txt')
dataset = SentencesPlusDocumentsDataset(sentences, tags, masks, document2sentences, sentence2position_in_document, tokenizer)
documents = Document(sentences, document2sentences, tokenizer)

print(dataset[13755])
print(dataset[6243])
print(dataset[2706])
print(dataset[7])

document_id = 200
words = documents.collect_all_positions_for_each_word(document_id)

"""
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([2.5, 2.2, 4.1, -1.0])
c = torch.tensor([1.0, 2.0, 3.0, 4.0])
d = torch.stack([a, b, c], dim=0)
print(d)
e = torch.mean(d, dim=0)
print(e)
print(e[0], e[1], e[2], e[3])"""
"""

for key in words:
    print(key, words[key]['bpe'])
    current_word = []
    for pos in words[key]['pos']:
        sentence_id = pos['sentence_id']
        if len(pos['ids']) == 1:
            position = pos['ids']
            print(sentence_id, position, documents[document_id][sentence_id][position])
            current_word.append(documents[document_id][sentence_id][position])
        else:
            position_start = pos['ids'][0]
            position_end = pos['ids'][-1]
            print(sentence_id, pos['ids'], documents[document_id][sentence_id][position_start: position_end + 1])
            current_word.append(documents[document_id][sentence_id][position_start: position_end + 1])
    all_context_vectors_of_a_word = torch.stack(current_word, dim=0)
    mean_context_vector_of_a_word = torch.mean(all_context_vectors_of_a_word, dim=0)
    
    words[key]['context_vector'] = mean_context_vector_of_a_word
    print('-' * 150)"""
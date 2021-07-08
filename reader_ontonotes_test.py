from ner.reader import ReaderOntonotes
from ner.document import Document

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

reader = ReaderOntonotes(include_document_ids=True)
sentences, documents_tags, documents_masks, document_to_sentences, sentences_to_documents_to_positions = reader.read('data/ontonotes/train')
documents = Document(sentences, document_to_sentences, tokenizer)
words = documents.collect_all_positions_for_each_word(0)
for word in words:
    print(word, words[word])
    if word == 100:
        break
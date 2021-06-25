from ner.reader import ReaderDocumentOntonotes

reader = ReaderDocumentOntonotes()
documents, documents_tags, documents_masks, document_id2sentences_ids = reader.get_sentences('data/ontonotes/train')
print(document_id2sentences_ids.keys())
from collections import Counter

from transformers import BertTokenizer
from ner.reader import ReaderCoNLL


def count_words_not_in_vocab(subset, sentences, labels, bert_vocab):
    entities_not_in_vocab = Counter()
    words_not_in_vocab = Counter()

    for sentence, sentence_labels in zip(sentences, labels):
        for word, label in zip(sentence, sentence_labels):
            if word not in bert_vocab:
                if label != 'O':
                    entities_not_in_vocab[word] += 1
                else:
                    words_not_in_vocab[word] += 1
    print(subset)
    print('Unique entities not in vocab {}'.format(len(entities_not_in_vocab.keys())))
    print('All entries of entities not in vocab {}'.format(sum(entities_not_in_vocab.values())))
    print('Unique words not in vocab {}'.format(len(words_not_in_vocab.keys())))
    print('All words of entities not in vocab {}'.format(sum(words_not_in_vocab.values())))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
bert_vocab = tokenizer.get_vocab().keys()

reader = ReaderCoNLL()
sentences, labels, masks = reader.read('data/conll2003/train.txt')

count_words_not_in_vocab('Train', sentences, labels, bert_vocab)

reader = ReaderCoNLL()
sentences, labels, masks = reader.read('data/conll2003/valid.txt')

count_words_not_in_vocab('Eval', sentences, labels, bert_vocab)

reader = ReaderCoNLL()
sentences, labels, masks = reader.read('data/conll2003/test.txt')

count_words_not_in_vocab('Test', sentences, labels, bert_vocab)
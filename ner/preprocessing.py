#https://github.com/daltonfury42/truecase

from typing import List
import truecase
import re

def truecase_sentence(tokens: List[str]) -> List[str]:
    word_list = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_list if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_list):
        parts = truecase.get_true_case(' '.join(lst)).split()

        # the truecase have its own tokenization ...
        # skip if the number of word doesn't match
        if len(parts) != len(word_list):
            return tokens

        for (w, idx), nw in zip(word_list, parts):
            tokens[idx] = nw

    return tokens
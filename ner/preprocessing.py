#https://github.com/daltonfury42/truecase

import truecase
import re


# original tokens
#['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']

def truecase_sentence(tokens):
   word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
   lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

   if len(lst) and len(lst) == len(word_lst):
       parts = truecase.get_true_case(' '.join(lst)).split()

       # the trucaser have its own tokenization ...
       # skip if the number of word dosen't match
       if len(parts) != len(word_lst): return tokens

       for (w, idx), nw in zip(word_lst, parts):
           tokens[idx] = nw

# truecased tokens
#['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']
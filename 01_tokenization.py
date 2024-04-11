#! pip install tokenizers

# https://huggingface.co/blog/how-to-train
# https://www.kaggle.com/code/abhishek/training-language-models-on-tpus-from-scratch/notebook
# https://www.youtube.com/watch?v=s-3zts7FTDA
# https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/bindings/python/py_src/tokenizers/implementations/bert_wordpiece.py#L1-L153
# https://huggingface.co/docs/tokenizers/pipeline#normalization

# pip install -e git+https://github.com/huggingface/transformers.git@main#egg=transformers[tensorflow]

from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from collections import defaultdict

paths = [str(x) for x in Path('./corpus').glob("**/*.txt")]

# print(paths)

bwpt = BertWordPieceTokenizer(
    vocab=None,
    # add_special_tokens=True,
    unk_token='[UNK]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True,
    wordpieces_prefix='##'
)
# paths=paths[0:5]

bwpt.train(
    files=paths,
    vocab_size=10000,
    min_frequency=2,
    limit_alphabet=1000,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]']
)

# Zapisz model i słownik
bwpt.save_model('./working')


# Pobranie słownika
vocab = bwpt.get_vocab()

# Utworzenie słownika z ilością wystąpień
word_counts = defaultdict(int)
for token, count in vocab.items():
    word_counts[token] = count

# Sortowanie słownika od największej liczby wystąpień
sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))

# Zapisz ilość wystąpień wyrazów do pliku
with open('./working/word_counts.txt', 'w', encoding='utf-8') as file:
    for token, count in sorted_word_counts.items():
        file.write(f'{token}: {count}\n')

# http://10.147.20.134:8888/tree?token=a29f558690009f8d290558ea9b5d9c716f74bee670f0d819
# https://docs.privategpt.dev/recipes
# https://huggingface.co/docs/transformers/main/en/model_doc/mistral
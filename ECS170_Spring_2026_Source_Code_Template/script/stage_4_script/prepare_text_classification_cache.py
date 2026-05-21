import argparse
import html
import os
import pickle
import re
from collections import Counter

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def tokenize(text):
    text = html.unescape(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    return TOKEN_RE.findall(text)


def read_split(data_dir, split):
    texts, labels = [], []
    for label_name, label_id in [('neg', 0), ('pos', 1)]:
        folder = os.path.join(data_dir, split, label_name)
        for file_name in sorted(os.listdir(folder)):
            if not file_name.endswith('.txt'):
                continue
            path = os.path.join(folder, file_name)
            with open(path, encoding='utf-8', errors='replace') as f:
                texts.append(f.read())
            labels.append(label_id)
    return texts, np.asarray(labels, dtype=np.int64)


def build_vocab(train_texts, vocab_size):
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))
    words = ['<pad>', '<unk>'] + [word for word, _ in counter.most_common(vocab_size - 2)]
    return {word: idx for idx, word in enumerate(words)}


def encode_texts(texts, vocab, max_length):
    unk_id = vocab['<unk>']
    encoded = np.zeros((len(texts), max_length), dtype=np.int64)
    for row, text in enumerate(texts):
        token_ids = [vocab.get(token, unk_id) for token in tokenize(text)[:max_length]]
        encoded[row, :len(token_ids)] = token_ids
    return encoded


def prepare_cache(data_dir, output_path, vocab_size, max_length):
    train_texts, train_y = read_split(data_dir, 'train')
    test_texts, test_y = read_split(data_dir, 'test')
    vocab = build_vocab(train_texts, vocab_size)

    cache = {
        'train': {
            'X': encode_texts(train_texts, vocab, max_length),
            'y': train_y,
        },
        'test': {
            'X': encode_texts(test_texts, vocab, max_length),
            'y': test_y,
        },
        'vocab': vocab,
        'vocab_size': len(vocab),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)

    print('saved cache:', output_path)
    print('train shape:', cache['train']['X'].shape)
    print('test shape:', cache['test']['X'].shape)
    print('vocab size:', cache['vocab_size'])


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    repo_root = os.path.abspath(os.path.join(project_root, '..'))
    data_dir = os.path.join(repo_root, 'stage_4_data', 'stage_4_data', 'text_classification')
    output_path = os.path.join(data_dir, 'processed_classification_cache_len128.pkl')

    parser = argparse.ArgumentParser(description='Prepare IMDb text classification cache for Stage 4.')
    parser.add_argument('--data-dir', default=data_dir)
    parser.add_argument('--output', default=output_path)
    parser.add_argument('--vocab-size', type=int, default=20000)
    parser.add_argument('--max-length', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare_cache(args.data_dir, args.output, args.vocab_size, args.max_length)

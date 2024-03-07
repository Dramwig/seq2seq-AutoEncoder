import pandas as pd
import torch
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as f:
        args_dict = yaml.safe_load(f)
    return argparse.Namespace(**args_dict)

def preprocess_X(X):
    # Add special tokens
    X = [['<'] + x + ['>'] for x in X]
    # Find maximum length
    max_len = max(len(x) for x in X)
    # Pad sequences
    X = [x + [''] * (max_len - len(x)) for x in X]
    return X, max_len

def build_vocab(X):
    # Building vocabulary
    special_tokens = ['', '<', '>', '?']
    vocab = torchtext.vocab.build_vocab_from_iterator(X, specials=special_tokens)
    vocab_dim = len(vocab)
    return vocab, vocab_dim

def build(sentence, vocab, max_len):
    tokens = list(sentence)
    tokens = ['<'] + tokens + ['>']
    tokens = tokens + [''] * (max_len - len(tokens))
    return [vocab[token] for token in tokens]

def rebuild(sentence, vocab):
    tokens = [vocab.get_itos()[token_idx] for token_idx in sentence]
    # Remove special tokens from the sentence
    tokens = [token for token in tokens if token not in ['', '<', '>', '?']]
    # Remove padding tokens
    return ''.join(tokens)

def load_dataset(batch_size, test_size=0.2):
    # Read the CSV file
    data = pd.read_csv('train.csv')
    X = data.iloc[:, 0].apply(list)
    
    # Preprocess X
    X, max_len = preprocess_X(X)
    data['url_tokens'] = X
    
    # Building vocabulary
    vocab, vocab_dim = build_vocab(X)
    data['ids'] = data['url_tokens'].apply(lambda tokens: vocab.lookup_indices(tokens))
    
    # Convert DataFrame columns to tensors
    X = list(data['ids'])
    Y = data['label'].values
    num_label = max(Y) + 1
    print(data.head())
    
    # Convert lists of indices to tensor and pad them
    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=vocab[''])
    Y = torch.tensor(Y)
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_padded, Y, test_size=test_size)
    
    # If X_train, X_test, y_train, y_test are already tensors.
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    # Convert to DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=False)
    
    print('train:', len(train_loader.dataset), 'test:', len(test_loader.dataset))
    print("train batch:", len(train_loader))
    return train_loader, test_loader, num_label, vocab, vocab_dim, data, max_len

# Example usage:
if __name__ == '__main__':
    batch_size = 32
    train_loader, test_loader, num_label, vocab, vocab_dim, data, max_len = load_dataset(batch_size)
    print(data['ids'][1])
    print(rebuild(data['ids'][1], vocab))
    print(build(rebuild(data['ids'][1], vocab), vocab, max_len))

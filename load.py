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

def build(sentence, vocab,):
    tokens = list(sentence)
    tokens = ['<'] + tokens + ['>']
    # tokens = tokens + [''] * (max_len - len(tokens))
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
    X = '<'+data['url']+'>'
    X  = [[c for c in str] for str in X]
    y = data['label'].values
        
    # Building vocabulary
    special_tokens = ['', '<', '>']
    vocab = torchtext.vocab.build_vocab_from_iterator(X, specials=special_tokens)
    vocab_dim = len(vocab)
    print("Vocabulary dimension:", vocab_dim)
    
    # 将字符转换为索引
    X_ids = [[vocab.get_stoi()[c] for c in url] for url in X]
        
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_ids, y, test_size=test_size)

    X_train = pad_sequence([torch.tensor(url) for url in X_train], batch_first=True)
    X_test = pad_sequence([torch.tensor(url) for url in X_test], batch_first=True)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    
    # If X_train, X_test, y_train, y_test are already tensors.
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    # Convert to DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=False)
    
    print('train:', len(train_loader.dataset), 'test:', len(test_loader.dataset))
    print("train batch:", len(train_loader))
    return train_loader, test_loader, vocab, vocab_dim, data

# Example usage:
if __name__ == '__main__':
    batch_size = 32
    train_loader, test_loader, vocab, vocab_dim, data = load_dataset(batch_size)
    for batch_idx, (src, trg) in enumerate(train_loader):
        print(src.T)
        break

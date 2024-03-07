# seq2seq

Minimal Seq2Seq model with attention for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.

This implementation relies on [torchtext](https://github.com/pytorch/text) to minimize dataset management and preprocessing parts.

## Model description



## Requirements

* GPU & CUDA
* Python3
* PyTorch
* torchtext
* Spacy
* numpy
* Visdom (optional)

download tokenizers by doing so:
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```


## References

Based on the following implementations

* [keon/seq2seq: Minimal Seq2Seq model with Attention for Neural Machine Translation in PyTorch (github.com)](https://github.com/keon/seq2seq)

- [bentrevett/pytorch-seq2seq: Tutorials on implementing a few sequence-to-sequence (seq2seq) models with PyTorch and TorchText. (github.com)](https://github.com/bentrevett/pytorch-seq2seq)

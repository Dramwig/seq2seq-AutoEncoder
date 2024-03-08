# seq2seq-AutoEncoder

A self-encoder written in pytorch for url dataset populating

Its purpose applies a DeepSMOTE-like approach to expand the string (URL) dataset to address the category imbalance problem. The aim here is to construct a sequence-to-sequence self-encoder and expand it using SMOTE. The code is still under construction ......

## Model description

The Seq2Seq model consists of three main components: the encoder, the decoder, and the Seq2Seq class that integrates them together.

The Encoder class takes an input sequence and produces a fixed-size representation of the input sequence. It consists of an embedding layer, which converts input tokens into dense vectors, and an LSTM layer, which processes the embedded tokens and outputs hidden states. The encoder's forward method takes the input sequence and returns the final hidden state and cell state.

The Decoder class generates the output sequence based on the hidden state and cell state from the encoder. It also uses an embedding layer and an LSTM layer. The forward method takes an input token, the hidden state, and the cell state, and returns the predicted output token, as well as the updated hidden state and cell state.

The Seq2Seq class combines the encoder and decoder. It takes the source sequence (input) and the target sequence (output) as input and performs the encoding and decoding steps. During decoding, it uses teacher forcing, which means it feeds the ground-truth tokens as inputs to the decoder with a certain probability.

## Installation

1. Clone the repository:

```
git clone https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion.git
```

2. Create a new Conda environment and activate it: 

```
conda env create -f environment.yml
conda activate pdiff
```

or install necessary package by:

```
pip install -r requirements.txt
```

### **Baseline**

```
python main.py
```

## References

Based on the following implementations

* [keon/seq2seq: Minimal Seq2Seq model with Attention for Neural Machine Translation in PyTorch (github.com)](https://github.com/keon/seq2seq)

- [bentrevett/pytorch-seq2seq: Tutorials on implementing a few sequence-to-sequence (seq2seq) models with PyTorch and TorchText. (github.com)](https://github.com/bentrevett/pytorch-seq2seq)
  # seq2seq-AutoEncoder
  
  

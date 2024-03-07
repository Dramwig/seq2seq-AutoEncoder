import torch
from model import Seq2Seq,Encoder,Decoder
from torch.utils.data import DataLoader, TensorDataset
from load import load_dataset, build, rebuild, load_arguments_from_yaml
import torch.nn.functional as F

def evaluate(model, test_loader, vocab_dim):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(test_loader):
            src = src.T
            src, trg = src.cuda(), src.cuda()
            output = model(src, trg)  
            
            # print(src)
            # print(src.shape)
            # output_2d = torch.argmax(output, dim=-1)
            # print(output_2d)
            # print(output_2d.shape)
            # input()
            
            loss = F.nll_loss(F.log_softmax(output.view(-1, vocab_dim),dim=1), trg.contiguous().view(-1), ignore_index = 0)
            total_loss += loss.item()  
        return total_loss / len(test_loader)

# Function to load the model
def load_model(model_path, model, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Assuming the `build` function correctly prepares your source sentence.
def translate_sentence(sentence, model, device, vocab, max_len):
    model.eval()
    print(sentence)

    # Convert the source sentence to indices and to tensor
    src_indices = build(sentence, vocab, max_len)
    src_tensor = torch.tensor(src_indices).unsqueeze(0)  # Add batch dimension
    src_tensor = src_tensor.to(device)  # Move to the correct device

    # No need for TensorDataset; directly use the tensor for inference
    with torch.no_grad():  # Do not compute gradients
        output = model(src_tensor, src_tensor)
        
    # 赋值给二维张量
    # 获取最后一维的最大值索引
    output_2d = torch.argmax(output, dim=-1)

    # Assuming `rebuild` converts model output indices back to a sentence
    translated_sentence = rebuild(torch.squeeze(output_2d, dim=0), vocab)
    print(translated_sentence)
    return translated_sentence


# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './.save/seq2seq_1_0.48.pt'
    filename = 'base.yaml'
    args = load_arguments_from_yaml(filename)
    
    train_loader, test_loader, vocab, vocab_dim, data = load_dataset(args.batch_size)
    encoder = Encoder(vocab_dim, args.embed_size, args.hidden_size,
                      n_layers=args.n_layers, dropout=args.encoder_dropout)
    decoder = Decoder(args.embed_size, args.hidden_size, vocab_dim,
                      n_layers=args.n_layers, dropout=args.decoder_dropout)
    seq2seq = Seq2Seq(encoder, decoder, device).cuda()
    
    model = load_model(model_path, seq2seq, device)
    loss = evaluate(model, test_loader, vocab_dim)
    print(f"Test loss: {loss:.4f}")

    while True:
        sentence = input("Please enter a sentence (exit): ")
        if sentence == "exit":
            break
        
        # sentence = data['url'][0]
        translate_sentence(sentence, model, device, vocab, max_len)

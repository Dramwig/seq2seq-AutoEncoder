import os, shutil
import math
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from load import load_dataset, load_arguments_from_yaml
from torch.optim.lr_scheduler import StepLR

def evaluate(model, test_loader, vocab_dim):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(test_loader):
            src, trg = src.cuda(), src.cuda()
            output = model(src, trg)  
            loss = F.nll_loss(F.log_softmax(output.view(-1, vocab_dim),dim=1), trg.contiguous().view(-1))
            total_loss += loss.item()  
        return total_loss / len(test_loader)

def train(model, optimizer, train_loader, vocab_dim, grad_clip):
    model.train()
    total_loss = 0
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.cuda(), src.cuda()  
        optimizer.zero_grad()
        output = model(src, trg)  
        loss = F.nll_loss(F.log_softmax(output.view(-1, vocab_dim),dim=1), trg.contiguous().view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()  

        if batch_idx!=0 and batch_idx % 10 == 0:  
            avg_loss = total_loss / 10
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                    (batch_idx, avg_loss, math.exp(avg_loss)))
            total_loss = 0


def main():
    filename = 'base.yaml'
    args = load_arguments_from_yaml(filename)
    print(args)
    
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    try:
        shutil.rmtree('.save')
    except:
        pass

    print("[!] preparing dataset...")
    train_loader, test_loader, num_label, vocab, vocab_dim, data, max_len = load_dataset(args.batch_size)
    
    print("[!] Instantiating models...")
    encoder = Encoder(vocab_dim, args.embed_size, args.hidden_size,
                      n_layers=args.n_layers, dropout=args.encoder_dropout)
    decoder = Decoder(args.embed_size, args.hidden_size, vocab_dim,
                      n_layers=args.n_layers, dropout=args.decoder_dropout)
    seq2seq = Seq2Seq(encoder, decoder, device).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(seq2seq, optimizer, train_loader, vocab_dim, args.grad_clip)
        val_loss = evaluate(seq2seq, test_loader, vocab_dim)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))
        
        # Updated learning rate.
        scheduler.step()

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            try:
                torch.save(seq2seq.cpu().state_dict(), './.save/seq2seq_%d_%.2f.pt' % (e, val_loss))
            except Exception as e:
                print(f"Error saving model: {e}")
            finally:
                seq2seq.cuda()
                best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

import torch
import torch.nn as nn
import argparse, os, time
import numpy as np
import random
import torch.optim as optim
from transformers import DistilBertModel, DistilBertConfig

from transformer.dataset import DataBase, pickleDataset
from transformer.Optim import ScheduledOptim
from transformer.Translator import Translator

# ---------------------------------------------------------------------------
# Adapter: wraps DistilBERT's encoder to match the interface Translator.py
# expects:  enc_output, *_ = self.model.encoder(src_seq, src_mask,
#                                                link=link, length=length)
# ---------------------------------------------------------------------------
class DistilBertEncoderAdapter(nn.Module):
    def __init__(self, distilbert):
        super().__init__()
        self.distilbert = distilbert  # the HuggingFace DistilBertModel

    def forward(self, src_seq, src_mask, link=None, length=None):
        # src_mask from Translator: [B, 1, seq_len]  (get_pad_mask output)
        # DistilBERT expects:       [B, seq_len]
        if src_mask.dim() == 3:
            attn_mask = src_mask.squeeze(1)          # [B, seq_len]
        else:
            attn_mask = src_mask

        # get_pad_mask returns 1 for REAL tokens, 0 for PAD
        # DistilBERT also expects 1=real, 0=pad — no inversion needed
        outputs = self.distilbert(
            input_ids      = src_seq,
            attention_mask = attn_mask,
        )
        enc_output = outputs.last_hidden_state   # [B, seq_len, d_model]
        return enc_output,                        # tuple so *_ unpacking works


# ---------------------------------------------------------------------------
# Adapter: a simple 1-layer decoder that attends to encoder output.
# Translator.py calls:
#   dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
# ---------------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, d_model, n_head, d_inner, n_layers, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model   = d_model,
            nhead     = n_head,
            dim_feedforward = d_inner,
            dropout   = dropout,
            batch_first = True,
        )
        self.decoder   = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.embedding = nn.Embedding(1, d_model)  # placeholder; real emb set externally

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        # trg_seq:   [B, trg_len]  — token ids
        # enc_output:[B, src_len, d_model]
        # src_mask:  [B, 1, src_len] or [B, src_len]

        tgt = self.embedding(trg_seq)              # [B, trg_len, d_model]

        # build memory_key_padding_mask from src_mask
        if src_mask.dim() == 3:
            mem_mask = (src_mask.squeeze(1) == 0)  # [B, src_len], True=ignore
        else:
            mem_mask = (src_mask == 0)

        dec_out = self.decoder(
            tgt                  = tgt,
            memory               = enc_output,
            memory_key_padding_mask = mem_mask,
        )
        return dec_out,   # tuple so *_ unpacking works


# ---------------------------------------------------------------------------
# Main model — exposes .encoder, .decoder, .trg_word_prj exactly as
# Translator.py expects
# ---------------------------------------------------------------------------
class DistilBertRuleformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx,
                 n_position, data, opt, nebor_relation):
        super().__init__()

        self.src_pad_idx    = src_pad_idx
        self.nebor_relation = nebor_relation
        self.opt            = opt

        d_model = opt.n_head * opt.d_v
        if d_model % opt.n_head != 0:
            d_model = (d_model // opt.n_head) * opt.n_head
        self.d_model = d_model

        # ---- DistilBERT encoder (random init — KG domain, not NLP) ----
        config = DistilBertConfig(
            vocab_size              = src_vocab_size,
            max_position_embeddings = max(n_position + 2, 512),
            dim                     = d_model,
            hidden_dim              = d_model * 4,
            n_heads                 = opt.n_head,
            n_layers                = opt.n_layers,
            dropout                 = opt.dropout,
            attention_dropout       = opt.dropout,
            pad_token_id            = src_pad_idx,
        )
        self.encoder = DistilBertEncoderAdapter(DistilBertModel(config))

        # ---- Decoder (standard Transformer decoder) ----
        self._decoder_core = SimpleDecoder(
            d_model  = d_model,
            n_head   = opt.n_head,
            d_inner  = d_model * 4,
            n_layers = max(1, opt.n_layers // 2),  # lighter decoder
            dropout  = opt.dropout,
        )
        # share the source embedding with the decoder token embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model,
                                          padding_idx=src_pad_idx)
        self._decoder_core.embedding = self.src_embedding

        # ---- Output projection (trg_word_prj) ----
        # Translator.py calls: self.model.trg_word_prj(dec_output)
        self.trg_word_prj = nn.Linear(d_model, trg_vocab_size, bias=False)

    # Translator.py calls self.model.decoder(...) — route to our adapter
    @property
    def decoder(self):
        return self._decoder_core

    def forward(self, input_ids):
        """Convenience forward used outside Translator (e.g. direct inference)."""
        attention_mask = (input_ids != self.src_pad_idx).long()
        enc_out, = self.encoder(input_ids, attention_mask.unsqueeze(1))
        return self.trg_word_prj(enc_out)   # [B, seq, trg_vocab]


# ---------------------------------------------------------------------------
def load_model(opt, device, nebor_relation):
    return DistilBertRuleformer(
        src_vocab_size = opt.src_vocab_size,
        trg_vocab_size = opt.trg_vocab_size,
        src_pad_idx    = opt.src_pad_idx,
        n_position     = opt.padding,
        data           = opt.data,
        opt            = opt,
        nebor_relation = nebor_relation.to(device),
    ).to(device)


# ---------------------------------------------------------------------------
def hit_mrr(hits, starttime):
    return (
        'MRR:{:.5f} @1:{:.5f} @3:{:.5f} @10:{:.5f} '
        'LOS:{:.5f} Time:{:.1f}secs'
    ).format(
        hits[10] / hits[12],
        hits[0]  / hits[12],
        hits[0:3].sum()  / hits[12],
        hits[0:10].sum() / hits[12],
        hits[11] / hits[12],
        time.time() - starttime,
    )


# ---------------------------------------------------------------------------
def run(translator, data_loader, id2r, mode, optimizer, device,
        padding, epoch, logfile, starttime, decode):

    hits = np.zeros(13)  # [0:10]=hit@k, [10]=mrr, [11]=loss, [12]=count

    for i, (subgraph, link, target, tailIndexs, length) in enumerate(data_loader):

        pred_seq, loss, indexL = translator(
            subgraph.to(device),
            target.to(device),
            tailIndexs.to(device),
            link.to(device),
            padding,
            mode,
            length.to(device),
        )

        if decode:
            continue

        print(f'\r {mode} {epoch}-{i}/{len(data_loader)}', end='    ')

        for index in indexL:
            if index < 10:
                hits[index] += 1
            hits[10] += 1 / (index + 1)
            hits[12] += 1
        hits[11] += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        print(hit_mrr(hits, starttime), end='       ')

    print(f'\r         {mode}-{epoch}  ' + hit_mrr(hits, starttime) + '     ')
    if logfile:
        with open(logfile, 'a') as log:
            log.write(f'{mode}-{epoch}  ' + hit_mrr(hits, starttime) + '\n')


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',       type=str,   required=True)
    parser.add_argument('-jump',       type=int,   required=True)
    parser.add_argument('-padding',    type=int,   required=True)
    parser.add_argument('-desc',       type=str,   required=True)
    parser.add_argument('-batch_size', type=int,   default=8)
    parser.add_argument('-epoch',      type=int,   default=10)
    parser.add_argument('-n_head',     type=int,   default=4)
    parser.add_argument('-d_v',        type=int,   default=32)
    parser.add_argument('-n_layers',   type=int,   default=6)
    parser.add_argument('-dropout',    type=float, default=0.1)
    parser.add_argument('-savestep',   type=int,   default=5)
    parser.add_argument('-ckpt',       type=str,   default='')
    parser.add_argument('-seed',       type=int,   default=31)
    parser.add_argument('-decode_rule',action='store_true', default=False)
    parser.add_argument('-the_rel',    type=float, default=0.6)
    parser.add_argument('-the_rel_min',type=float, default=0.3)
    parser.add_argument('-the_all',    type=float, default=0.1)

    opt = parser.parse_args()

    opt.d_k         = opt.d_v
    opt.d_model     = opt.n_head * opt.d_v
    opt.d_word_vec  = opt.d_model
    opt.d_inner_hid = opt.d_model * 4
    opt.subgraph    = opt.data + f'/subgraph{opt.jump}'

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    base_data = DataBase('DATASET/' + opt.data, subgraph='DATASET/' + opt.subgraph)
    opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()
    opt.src_pad_idx = base_data.e2id['<pad>']   # never hardcode 0

    train_data  = pickleDataset(base_data, opt, mode='train')
    valid_data  = pickleDataset(base_data, opt, mode='valid')
    test_data   = pickleDataset(base_data, opt, mode='test')

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size,
        shuffle=True,  collate_fn=pickleDataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=opt.batch_size,
        shuffle=False, collate_fn=pickleDataset.collate_fn)
    test_loader  = torch.utils.data.DataLoader(
        test_data,  batch_size=opt.batch_size,
        shuffle=False, collate_fn=pickleDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] Device  : {device}')
    print(f'[Info] d_model={opt.d_model}, n_head={opt.n_head}, n_layers={opt.n_layers}')

    translator = Translator(
        model     = load_model(opt, device, base_data.nebor_relation),
        opt       = opt,
        device    = device,
        base_data = base_data,
    ).to(device)

    if opt.ckpt:
        translator.load_state_dict(
            torch.load(opt.ckpt, map_location=device), strict=False)
        print(f'[Info] Loaded checkpoint: {opt.ckpt}')

    os.makedirs(f'EXPS/{opt.desc}', exist_ok=True)
    logfile   = f'EXPS/{opt.desc}/log.txt'
    starttime = time.time()

    optimizer = ScheduledOptim(
        optim.Adam(translator.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, 400,
    )

    for epoch in range(opt.epoch):
        run(translator, train_loader, base_data.id2r, 'train',
            optimizer, device, opt.padding, epoch+1, logfile, starttime, False)

        if opt.savestep and (epoch + 1) % opt.savestep == 0:
            ckpt = f'EXPS/{opt.desc}/model_epoch_{epoch+1}.pt'
            torch.save(translator.state_dict(), ckpt)
            print(f'[Info] Saved {ckpt}')

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                run(translator, valid_loader, base_data.id2r, 'valid',
                    optimizer, device, opt.padding, epoch+1, logfile, starttime, False)
                run(translator, test_loader, base_data.id2r, 'test',
                    optimizer, device, opt.padding, epoch+1, logfile, starttime, False)

    print('\n[Info] Training complete.')


if __name__ == '__main__':
    main()

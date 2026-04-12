import torch
import torch.nn as nn
import argparse, os, time
from transformers import DistilBertModel, DistilBertConfig
from transformer.Translator import Translator
from transformer.dataset import DataBase, pickleDataset
from transformer.Optim import ScheduledOptim
import torch.optim as optim
import numpy as np
import random


class DistilBertRuleformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx,
                 n_position, data, opt, nebor_relation):
        super().__init__()

        self.src_pad_idx = src_pad_idx

        d_model = opt.n_head * opt.d_v
        if d_model % opt.n_head != 0:
            d_model = (d_model // opt.n_head) * opt.n_head

        config = DistilBertConfig(
            vocab_size=src_vocab_size,
            max_position_embeddings=max(n_position + 2, 512),
            dim=d_model,
            hidden_dim=d_model * 4,
            n_heads=opt.n_head,
            n_layers=opt.n_layers,
            dropout=opt.dropout,
            attention_dropout=opt.dropout,
            pad_token_id=src_pad_idx,
        )

        self.encoder = DistilBertModel(config)
        self.output_proj = nn.Linear(d_model, trg_vocab_size)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.src_pad_idx).long()

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return self.output_proj(outputs.last_hidden_state)


def load_model(opt, device, nebor_relation):
    model = DistilBertRuleformer(
        src_vocab_size=opt.src_vocab_size,
        trg_vocab_size=opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        n_position=opt.padding,
        data=opt.data,
        opt=opt,
        nebor_relation=nebor_relation.to(device),
    ).to(device)
    return model


def run(translator, data_loader, id2r, mode, optimizer, device,
        padding, epoch, logfile, starttime, decode):

    loss_fn = nn.CrossEntropyLoss(ignore_index=translator.model.src_pad_idx)

    for i, (_, _, target, _, _) in enumerate(data_loader):

        target = target.to(device)

        input_ids = target[:, :-1]
        labels = target[:, 1:]

        logits = translator.model(input_ids)

        labels = labels.clone()
        labels[labels >= logits.size(-1)] = translator.model.src_pad_idx
        labels[labels < 0] = translator.model.src_pad_idx

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        print(f'\r {mode} {epoch}-{i}/{len(data_loader)} loss:{loss.item():.4f}', end='')

        if mode == 'train':
            loss.backward()
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-jump', type=int, required=True)
    parser.add_argument('-padding', type=int, required=True)
    parser.add_argument('-desc', type=str, required=True)

    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-d_v', type=int, default=32)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-decode_rule', action='store_true', default=False)
    parser.add_argument('-the_rel', type=float, default=0.6)
    parser.add_argument('-the_rel_min', type=float, default=0.3)
    parser.add_argument('-the_all', type=float, default=0.1)

    opt = parser.parse_args()

    opt.src_pad_idx = 0

    base_data = DataBase(opt.data, subgraph=opt.data + f'/subgraph{opt.jump}')
    opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()

    train_data = pickleDataset(base_data, opt, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=pickleDataset.collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(opt, device, base_data.nebor_relation)

    translator = Translator(
        model=model,
        opt=opt,
        device=device,
        base_data=base_data
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(translator.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.n_head * opt.d_v, 400
    )

    for epoch in range(opt.epoch):
        run(translator, train_loader, base_data.id2r, 'train',
            optimizer, device, opt.padding, epoch + 1, "", time.time(), False)
    
    print("\nTraining Complete")


if __name__ == "__main__":
    main()


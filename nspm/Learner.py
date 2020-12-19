from torchtext.data import Field, BucketIterator, TabularDataset
from utils import load_checkpoint, save_checkpoint
from torchtext.data.utils import get_tokenizer
from spacy.tokenizer import Tokenizer
import torch
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.seq_to_seq_bilstm import BiLSTM
from models.seq_to_seq_transformer import Transformer
import matplotlib.pyplot as plt


class Learner():
    def __init__(self, train_filename, test_filename,
                 num_epochs, batch_size, learning_rate,
                 max_len, device):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.device = device

        eng_tokenizer = get_tokenizer('basic_english')
        sql_tokenizer = lambda x: x.split(' ')

        self.english = Field(sequential=True, tokenize=eng_tokenizer, lower=True,
                             init_token="<sos>", eos_token="<eos>")
        self.sparql = Field(sequential=True, tokenize=sql_tokenizer, lower=True,
                            init_token="<sos>", eos_token="<eos>")

        fields = [('src', self.english), ('trg', self.sparql)]

        self.train_data, self.test_data = TabularDataset.splits(
            path='',
            train=train_filename,
            test=test_filename,
            format='csv',
            skip_header=True,
            fields=fields
        )

    def build_vocab(self):
        self.english.build_vocab(self.train_data, max_size=20000)
        self.sparql.build_vocab(self.train_data, max_size=20000)
        return self.english, self.sparql

    def get_data(self):
        return self.train_data, self.test_data


    def train_transformer_model(self,
                                embedding_size,
                                num_heads,
                                enc_nlayers,
                                dec_nlayers,
                                forward_expansion,
                                dropout,
                                load_model, save_model=True,
                                checkpoint_name="models/tfmr_chkpt.pth.tar"):

        src_pad_idx = self.english.vocab.stoi['<pad>']
        src_vocab_size = len(self.english.vocab)
        trg_vocab_size = len(self.sparql.vocab)

        model = Transformer(
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            enc_nlayers,
            dec_nlayers,
            forward_expansion,
            dropout,
            self.max_len,
            self.device
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
        criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


        # Tensorboard to get nice loss plot
        writer = SummaryWriter(f"runs/loss_plot_")
        step = 0

        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device,
        )


        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        for epoch in range(self.num_epochs):
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, checkpoint_name)

            model.train()

            losses = []
            for batch_idx, batch in enumerate(train_iterator):
                src_data = batch.src.to(self.device)
                trg_data = batch.trg.to(self.device)

                # Forward prop
                output = model(src_data, trg_data[:-1])
                output = output.reshape(-1, trg_vocab_size)

                trg_data = trg_data[1:].reshape(-1)

                optimizer.zero_grad()

                loss = criterion(output, trg_data)
                losses.append(loss.item())

                loss.backward()

                # Clip to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                # Gradient descent step
                optimizer.step()

                # Plot to tensorboard
                writer.add_scalar("Training loss", loss, global_step=step)
                step += 1
            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            print(f"[epoch: {epoch} / {self.num_epochs} | loss: {mean_loss}]")





    def train_bilstm_model(self,
                           embedding_size,
                           hidden_size,
                           nlayers,
                           dropout,
                           load_model, save_model=True,
                           checkpoint_name="models/blstm_chkpt.pth.tar"):

        src_pad_idx = self.english.vocab.stoi['<pad>']

        model = BiLSTM(len(self.english.vocab),
                       len(self.sparql.vocab),
                       embedding_size,
                       hidden_size,
                       len(self.sparql.vocab),
                       nlayers,
                       dropout,
                       self.device,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
        criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


        writer = SummaryWriter(f"runs/loss_plot_")
        step = 0

        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device,
        )

        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        for epoch in range(self.num_epochs):
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, checkpoint_name)

            model.train()

            losses = []
            for batch_idx, batch in enumerate(train_iterator):
                src_data = batch.src.to(self.device)
                trg_data = batch.trg.to(self.device)

                # Forward prop
                output = model(src_data, trg_data)

                output = output[1:].reshape(-1, output.shape[2])
                trg_data = trg_data[1:].reshape(-1)

                optimizer.zero_grad()
                loss = criterion(output, trg_data)
                losses.append(loss.item())

                loss.backward()

                # Clip to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                # Gradient descent step
                optimizer.step()

                # Plot to tensorboard
                writer.add_scalar("Training loss", loss, global_step=step)
                step += 1
            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            print(f"[epoch: {epoch} / {self.num_epochs} | loss: {mean_loss}]")

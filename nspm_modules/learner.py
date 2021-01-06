from torchtext.data import Field, BucketIterator, TabularDataset
from utils import load_checkpoint, save_checkpoint
from torchtext.data.utils import get_tokenizer
import torch
from utils import bleu
from torch import optim
import torch.nn as nn
from models.seq_to_seq_bilstm import BiLSTM
from models.seq_to_seq_transformer import Transformer
from time import time


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
            train="data/train_test_files/" + train_filename,
            test="data/train_test_files/" + test_filename,
            format='csv',
            skip_header=True,
            fields=fields
        )

        self.english.build_vocab(self.train_data, max_size=20000, min_freq=2)
        self.sparql.build_vocab(self.train_data, max_size=20000, min_freq=2)

        self.src_pad_idx = self.english.vocab.stoi['<pad>']
        self.src_vocab_size = len(self.english.vocab)
        self.trg_vocab_size = len(self.sparql.vocab)


        self.train_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device,
        )

    def get_vocab(self):
        return self.english, self.sparql

    def train_transformer_model(self,
                                embedding_size,
                                num_heads,
                                enc_nlayers,
                                dec_nlayers,
                                forward_expansion,
                                dropout,
                                load_model, save_model=True,
                                checkpoint_name="models/tfmr_wd_chkpt.pth.tar"):

        print("\nTraining Transformer model...\n")
        model = Transformer(embedding_size, self.src_vocab_size, self.trg_vocab_size, self.src_pad_idx,
                            num_heads, enc_nlayers, dec_nlayers, forward_expansion,
                            dropout, self.max_len, self.device
                ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)

        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        t0 = time()
        for epoch in range(self.num_epochs):
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, checkpoint_name)

            model.train()
            losses = []
            for batch_idx, batch in enumerate(self.train_iterator):
                src_data = batch.src.to(self.device)
                trg_data = batch.trg.to(self.device)

                output = model(src_data, trg_data[:-1])
                output = output.reshape(-1, self.trg_vocab_size)
                trg = trg_data[1:].reshape(-1)

                optimizer.zero_grad()

                loss = criterion(output, trg)
                losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()

            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            print(f"[epoch: {epoch+1}/{self.num_epochs} | loss: {mean_loss}] elapsed: {(time()-t0)//60} min")


    def test_transformer_model(self,
                                embedding_size,
                                num_heads,
                                enc_nlayers,
                                dec_nlayers,
                                forward_expansion,
                                dropout,
                                load_model,
                                checkpoint_name="models/tfmr_chkpt.pth.tar"):

        print("\nTesting Transformer model...\n")
        model = Transformer(embedding_size, self.src_vocab_size, self.trg_vocab_size, self.src_pad_idx,
                            num_heads, enc_nlayers, dec_nlayers, forward_expansion,
                            dropout, self.max_len, self.device
                ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)

        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        with torch.no_grad():
            model.eval()

            targets, predictions = [], []
            losses = []
            for _, batch in enumerate(self.test_iterator):
                src_data = batch.src.to(self.device)
                trg_data = batch.trg.to(self.device)

                output = model(src_data, trg_data[:-1])
                output = output.reshape(-1, self.trg_vocab_size)
                trg = trg_data[1:].reshape(-1)

                targets.append([trg_data])
                predictions.append(output)

                loss = criterion(output, trg)
                losses.append(loss.item())

            print(f"Loss: {sum(losses) / len(losses)}")

            bleu_score, acc = bleu(self.test_data[1:], model, self.english, self.sparql, self.device)
            print(f"Bleu score: {bleu_score * 100:.2f} | Accuracy: {acc * 100:.2f}")



    def train_bilstm_model(self,
                           embedding_size,
                           hidden_size,
                           nlayers,
                           dropout,
                           load_model, save_model=True,
                           checkpoint_name="models/blstm_chkpt.pth.tar"):

        print("\nTraining BiLSTM model...\n")
        model = BiLSTM(len(self.english.vocab), len(self.sparql.vocab),
                       embedding_size, hidden_size, nlayers, dropout, self.device,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)

        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device,
        )

        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        t0 = time()
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

                output = model(src_data, trg_data)
                output = output[1:].reshape(-1, output.shape[2])
                trg = trg_data[1:].reshape(-1)

                optimizer.zero_grad()

                loss = criterion(output, trg)
                losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()

            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            print(f"[epoch: {epoch+1}/{self.num_epochs} | loss: {mean_loss}] elapsed: {(time()-t0)//60} min")


    def test_bilstm_model(self,
                           embedding_size,
                           hidden_size,
                           nlayers,
                           dropout,
                           load_model,
                           checkpoint_name="models/blstm_chkpt.pth.tar"):

        print("\nTesting BiLSTM model...\n")
        model = BiLSTM(len(self.english.vocab), len(self.sparql.vocab),
                       embedding_size, hidden_size, nlayers, dropout, self.device,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)


        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        with torch.no_grad():
            model.eval()

            targets, predictions = [], []
            losses = []
            for _, batch in enumerate(self.test_iterator):
                src_data = batch.src.to(self.device)
                trg_data = batch.trg.to(self.device)

                output = model(src_data, trg_data)
                output = output[1:].reshape(-1, output.shape[2])
                trg = trg_data[1:].reshape(-1)

                targets.append([trg])
                predictions.append(output)

                loss = criterion(output, trg)
                losses.append(loss.item())

            print(f"Loss: {sum(losses) / len(losses)}")

            bleu_score, accuracy = bleu(self.test_data[1:], model, self.english, self.sparql, self.device)
            print(f"Bleu score: {bleu_score * 100:.2f} | Accuracy: {accuracy * 100:.2f}")

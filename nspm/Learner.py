from torchtext.data import Field, BucketIterator, TabularDataset
from utils import load_checkpoint, save_checkpoint
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Learner():
    def __init__(self, train_filename, test_filename):
        self.english = Field(sequential=True, tokenize='spacy', lower=True,
                             init_token="<sos>", eos_token="<eos>")
        self.sparql = Field(sequential=True, tokenize='spacy', lower=True,
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


    def build_vocabs(self):
        self.english.build_vocab(self.train_data, max_size=20000, min_freq=2)
        self.sparql.build_vocab(self.train_data, max_size=20000, min_freq=2)
        return self.english, self.sparql


    def train_model(self, model, optimizer,
                    device,
                    batch_size, num_epochs,
                    checkpoint_name, load_model, save_model=True):

        # Tensorboard to get nice loss plot
        writer = SummaryWriter(f"runs/loss_plot_")
        step = 0

        train_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.test_data),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )

        pad_idx = self.english.vocab.stoi["<pad>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        if load_model:
            load_checkpoint(torch.load(checkpoint_name), model, optimizer)

        for epoch in range(num_epochs):
            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, checkpoint_name)

            model.train()

            losses = []
            for batch_idx, batch in enumerate(train_iterator):
                src_data = batch.src.to(device)
                trg_data = batch.trg.to(device)

                # Forward prop
                output = model(src_data, trg_data)

                output = output[1:].reshape(-1, output.shape[2])
                trg_data = trg_data[1:].reshape(-1)

                optimizer.zero_grad()
                loss = criterion(output, trg_data)
                losses.append(loss.item())

                print(f"[epoch: {epoch} / {num_epochs}, loss: {loss}]")
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
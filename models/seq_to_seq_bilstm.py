import torch
import torch.nn as nn
import random


class BiLSTM(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 device,
                 teacher_force_ratio=0.4):
        super(BiLSTM, self).__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embedding_size, hidden_size,
                                   num_layers, dropout)
        self.decoder = LSTMDecoder(trg_vocab_size, embedding_size, hidden_size,
                                   trg_vocab_size, num_layers, dropout)
        self.device = device
        self.trg_vocab_size = trg_vocab_size
        self.teacher_force_ratio = teacher_force_ratio

    def forward(self, src, trg):
        trg_seq_length, N = trg.shape

        outputs = torch.zeros(trg_seq_length, N, self.trg_vocab_size).to(self.device)
        encoder_states, hidden, cell = self.encoder(src)

        x = trg[0]

        for t in range(1, trg_seq_length):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output
            best_guess = output.argmax(1)

            x = trg[t] if random.random() < self.teacher_force_ratio else best_guess
        return outputs


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        encoder_states, (hidden, cell) = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, input_size,
                 embedding_size,
                 hidden_size,
                 output_size,
                 num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell


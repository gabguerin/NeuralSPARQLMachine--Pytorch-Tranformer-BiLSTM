from NSpM.Generator import Generator
from NSpM.Learner import Learner
from models.seq_to_seq_bilstm import BiLSTM
import pandas as pd
import torch
from torch import optim
import torch.nn as nn


lcquad_train = pd.read_json("data/LC-QuAD2.0/dataset/train.json")[["question","sparql_wikidata"]]
lcquad_test = pd.read_json("data/LC-QuAD2.0/dataset/test.json")[["question","sparql_wikidata"]]

data = pd.concat([lcquad_train, lcquad_test], ignore_index=True).dropna()
nl_questions = data.question.values
wd_queries = data.sparql_wikidata.values

"""
    Build the Generator to generate files with form:
        question,                   sparql_query
        what medical specialty...., select var1 where brack_open wd_q168805...
"""
generator = Generator(nl_questions, wd_queries)

train_filename, test_filename = "lcquad_train.csv", "lcquad_test.csv"
generator.generate_train_test_files(train_filename, test_filename)


"""
    Build the Learner to get the input and target vocabs
"""
learner = Learner(train_filename, test_filename)
english, sparql = learner.build_vocabs()


"""
    Hyperparameters to train the seq_to_seq_bilstm model
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
load_model = False
save_model = True
checkpoint_name = "models/nspm_chkpt.pth.tar"

num_epochs = 100
learning_rate = 1e-3
batch_size = 16

input_size_enc = len(english.vocab)
input_size_dec = len(sparql.vocab)
output_size = len(sparql.vocab)
embedding_size = 128
hidden_size = 512
num_layers = 2
dropout = 0.1

"""
    Build & Train the model
"""
model = BiLSTM(input_size_enc, input_size_dec,
                embedding_size,
                hidden_size,
                output_size,
                num_layers, dropout,
                device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


learner.train_model(model, optimizer,
                    device,
                    batch_size, num_epochs,
                    checkpoint_name, load_model, save_model=True)


import warnings
#warnings.filterwarnings("ignore")
from nspm.Generator import Generator
from nspm.Learner import Learner
import pandas as pd
import torch
from torch import optim
from models.seq_to_seq_bilstm import BiLSTM
from utils import load_checkpoint, save_checkpoint, query_prediction


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
    Hyperparameters to train the seq_to_seq_bilstm model
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 50
learning_rate = 1e-3
batch_size = 16
max_len = 300

"""
    Build the Learner to get the input and target vocabs
"""
learner = Learner(train_filename, test_filename,
                  num_epochs, batch_size, learning_rate,
                  max_len, device)
english, sparql = learner.build_vocab()

trainning = [None,"transformer","bilstm"][1]
load_model = True

"""
    Build & Train the Transformer model
"""
embedding_size = 256
num_heads = 8
enc_nlayers = 3
dec_nlayers = 3
dropout = 0.1
forward_expansion = 4

if trainning == "transformer":
    learner.train_transformer_model(embedding_size,
                                    num_heads,
                                    enc_nlayers,
                                    dec_nlayers,
                                    forward_expansion,
                                    dropout,
                                    load_model)

"""
    Build & Train the BiLSTM model
"""

embedding_size = 128
hidden_size = 512
nlayers = 1
dropout = 0.1

if trainning == "bilstm":
    learner.train_bilstm_model(embedding_size,
                               hidden_size,
                               nlayers,
                               dropout,
                               load_model)

"""
    Evaluating models
"""

model = BiLSTM(len(english.vocab),
               len(sparql.vocab),
               embedding_size,
               hidden_size,
               len(sparql.vocab),
               nlayers,
               dropout,
               device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model, optimizer = load_checkpoint(torch.load("blstm_chkpt.pth.tar"), model, optimizer)

print(query_prediction(model, "tell me the name of solstice which starts with s", english, sparql, device, max_length=50))

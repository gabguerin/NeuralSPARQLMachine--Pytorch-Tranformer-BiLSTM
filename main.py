from nspm_modules.Generator import Generator
from nspm_modules.Learner import Learner
import pandas as pd
import torch


lcquad_train = pd.read_json("data/LC-QuAD2.0/dataset/train.json")[["question","sparql_wikidata"]]
lcquad_test = pd.read_json("data/LC-QuAD2.0/dataset/test.json")[["question","sparql_wikidata"]]

data = pd.concat([lcquad_train, lcquad_test], ignore_index=True).dropna()
nl_questions = data.question.values
wd_queries = data.sparql_wikidata.values

"""
    Build the Generator to generate pairs under the form:
        question :  what is philippine standard geographic code for angeles
        query :     select distinct var1 where bkt_open wd_qxxx wdt_pxxx var1 bkt_close
"""
generator = Generator(nl_questions, wd_queries)

train_filename, test_filename = "lcquad_train.csv", "lcquad_test.csv"
#generator.generate_train_test_files(train_filename, test_filename)


"""
    Hyperparameters to train the seq_to_seq_bilstm model
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
learning_rate = 1e-3
batch_size = 16
max_len = 300

"""
    Build the Learner to get the input and target vocabs
"""
learner = Learner(train_filename, test_filename,
                  num_epochs, batch_size, learning_rate,
                  max_len, device)



training = [None,"transformer","bilstm"][2]
testing = [None,"transformer","bilstm"][2]
load_model = True

"""
    Build & Train the Transformer model
"""
embedding_size = 128
num_heads = 8
enc_nlayers = 3
dec_nlayers = 3
dropout = 0.1
forward_expansion = 4

if training == "transformer":
    learner.train_transformer_model(embedding_size,
                                    num_heads,
                                    enc_nlayers,
                                    dec_nlayers,
                                    forward_expansion,
                                    dropout,
                                    load_model)

if testing == "transformer":
    learner.test_transformer_model(embedding_size,
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
hidden_size = 64
nlayers = 1
enc_dropout = 0.1
dec_dropout = 0.1

if training == "bilstm":
    learner.train_bilstm_model(embedding_size,
                                   hidden_size,
                                   nlayers,
                                   dropout,
                                   load_model)

if testing == "bilstm":
    learner.test_bilstm_model(embedding_size,
                                   hidden_size,
                                   nlayers,
                                   dropout,
                                   load_model)



"""
    Build the Interpreter to get the corrected SPARQL query
"""
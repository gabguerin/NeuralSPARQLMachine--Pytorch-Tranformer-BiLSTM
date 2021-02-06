from nspm_modules.generator import Generator
from nspm_modules.learner import Learner
from nspm_modules.interpreter import Interpreter
from models.seq_to_seq_transformer import Transformer
from torch import optim
from utils import load_checkpoint
import pandas as pd
import torch



generate_files = False
training = [None,"transformer","bilstm"][1]
testing = [None,"transformer","bilstm"][1]
load_model = False
train_filename, test_filename = "lcquad_wd_train.csv", "lcquad_wd_test.csv"
checkpoint_name = "models/tfmr_wd_chkpt.pth.tar"


"""
    Build the Generator to generate pairs under the form:
        question :  what is philippine standard geographic code for angeles
        query :     select distinct var1 where bkt_open wd_qxxx wdt_pxxx var1 bkt_close
"""

if generate_files:
    lcquad_train = pd.read_json("data/LC-QuAD2.0/dataset/train.json")[["question", "sparql_wikidata"]]
    lcquad_test = pd.read_json("data/LC-QuAD2.0/dataset/test.json")[["question", "sparql_wikidata"]]

    data = pd.concat([lcquad_train, lcquad_test], ignore_index=True).dropna()
    nl_questions = data["question"].values
    spl_queries = data["sparql_wikidata"].values

    generator = Generator(nl_questions, spl_queries)

    generator.generate_train_test_files(train_filename, test_filename)


"""
    Hyperparameters to train the seq_to_seq model
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 200
learning_rate = 1e-2
batch_size = 16
max_len = 300

"""
    Build the Learner
"""
learner = Learner(train_filename, test_filename,
                  num_epochs, batch_size, learning_rate,
                  max_len, device)


"""
    Build & Train the Transformer model
"""
embedding_size = 1024
num_heads = 8
enc_nlayers = 4
dec_nlayers = 4
dropout = 0.1
forward_expansion = 4

if training == "transformer":
    learner.train_transformer_model(embedding_size,
                                    num_heads,
                                    enc_nlayers,
                                    dec_nlayers,
                                    forward_expansion,
                                    dropout,
                                    load_model,
                                    checkpoint_name=checkpoint_name)

if testing == "transformer":
    learner.test_transformer_model(embedding_size,
                                    num_heads,
                                    enc_nlayers,
                                    dec_nlayers,
                                    forward_expansion,
                                    dropout,
                                    load_model,
                                   checkpoint_name=checkpoint_name)


"""
    Build & Train the BiLSTM model
"""
"""
embedding_size = 128
hidden_size = 64
nlayers = 1
enc_dropout = 0.1
dec_dropout = 0.1

if training == "bilstm":
    learner.train_bilstm_model(embedding_size, hidden_size, nlayers, dropout, load_model)

if testing == "bilstm":
    learner.test_bilstm_model(embedding_size, hidden_size, nlayers, dropout, load_model)

"""


"""
    Build the Interpreter to get the corrected SPARQL query
"""
english, sparql = learner.get_vocab()

model = Transformer(embedding_size,
                    len(english.vocab), len(sparql.vocab), english.vocab.stoi['<pad>'],
                    num_heads, enc_nlayers, dec_nlayers,
                    forward_expansion,
                    dropout,
                    max_len,
                    device).to(device)
load_checkpoint(torch.load(checkpoint_name), model, optim.Adam(model.parameters(), lr=learning_rate))

interpreter = Interpreter(model, english, sparql, device)

questions = ["what award did Bob Cousy receive in 1961 ?",
             "what is the sexual orientation and the gender of Ricky Martin ?",
             "Which is the institution that contains the word roman in their name ?",
             "Do the points for Chris Harper equal 10 ?",
             "What nba team starts with the letter D ?",
             "When did the father of Leonardo Da Vinci die ?"]

for question in questions:
    print("<Interpreter input>")
    print(question)
    print("<Interpreter output>")
    print(interpreter.query_from_question(question))

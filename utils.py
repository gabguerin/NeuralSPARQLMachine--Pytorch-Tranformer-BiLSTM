import torch
from torchtext.data.utils import get_tokenizer
import difflib
from torchtext.data.metrics import bleu_score

def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def query_prediction(model, tokens, english, sparql, device, max_length=100):
    # Add <SOS> and <EOS> tokens
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    text_to_indices = [english.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [sparql.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == sparql.vocab.stoi["<eos>"]:
            break

    query_pred = [sparql.vocab.itos[idx] for idx in outputs]
    return query_pred[1:]


def bleu(data, model, english, sparql, device):
    targets = []
    outputs = []

    acc = 0
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = query_prediction(model, src, english, sparql, device)
        prediction = prediction[:-1]

        targets.append([trg])
        outputs.append(prediction)

        acc += difflib.SequenceMatcher(None, trg, prediction).ratio()

    return bleu_score(outputs, targets), acc / len(data)


from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
from bs4 import BeautifulSoup
import json

def get_query_answer(queryString):
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql",
                           agent="answering-complex-questions 0.1 (github.com/D2KLab/AnsweringComplexQuestions)")
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    #print(sparql)
    results = sparql.query().convert()
    #print(results)
    if 'results' in results.keys() and len((results['results']['bindings'])) > 0:
        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        col_value = results_df.columns[-1]
        return results_df[col_value][0]
    elif 'boolean' in results.keys():
        return results['boolean']
    return 'Not Found'


def get_label_from_id(entity):
    response = urllib.request.urlopen("http://www.wikidata.org/entity/"+entity)
    html_doc = response.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    data = json.loads(str(soup))
    label = data["entities"][entity]['labels']['en']['value']
    if label.endswith("T00:00:00Z"):
        return label.split('T00:00:00Z')[0]
    return label

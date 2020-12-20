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


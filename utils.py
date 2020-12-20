import torch
from torchtext.data.utils import get_tokenizer

def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def query_prediction(model, sentence, english, sparql, device, max_length=50):
    tokenizer = get_tokenizer('basic_english')

    if type(sentence) != str:
        tokens = [token.text.lower() for token in tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
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
    # remove start token
    return query_pred[1:]


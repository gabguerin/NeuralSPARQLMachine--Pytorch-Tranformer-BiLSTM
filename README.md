# NeuralSPARQLMachine--Pytorch-Tranformer-BiLSTM
### The architecture of this Neural SPARQL Machine is inspired by the one below:
<p align="center">
  <img src="https://github.com/gabguerin/NeuralSPARQLMachine--Pytorch-Tranformer-BiLSTM/blob/main/data/NSpM.PNG" width="450" height="400">
</p>

## 1. Encode data (Generator)
```bash
Original data:
      question : What is the ISSF ID of Kim Rhode?
  sparql_query : select distinct ?answer where { wd:Q233759 wdt:P2730 ?answer}

Generator output:
      question : what is the issf id of kim rhode
  sparql_query : select distinct var1 where bkt_open wd_qxxx wdt_pxxx var1 bkt_close
```

## 2. Train & Test transformer model (Learner)

## 3. Decode data & Entity linking (Interpreter)

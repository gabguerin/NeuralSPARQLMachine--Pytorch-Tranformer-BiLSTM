# NeuralSPARQLMachine--Pytorch-Tranformer-BiLSTM
> The architecture of this Neural SPARQL Machine is inspired by the one prensented in this [paper](https://s3.eu-west-2.amazonaws.com/tsoru.aksw.org/neural-sparql-machines/soru-marx-semantics2017.html) :
<p align="center">
  <img src="https://github.com/gabguerin/NeuralSPARQLMachine--Pytorch-Tranformer-BiLSTM/blob/main/data/NSpM.PNG" width="450" height="400">
</p>

## 1. Encode data (Generator)
```sparql
<Generator input>
      question : What is the ISSF ID of Kim Rhode?
  sparql_query : SELECT DISCTINCT ?answer WHERE { wd:Q233759 wdt:P2730 ?answer}

<Generator output>
      question : what is the issf id of kim rhode
  sparql_query : select distinct var1 where bkt_open wd_qxxx wdt_pxxx var1 bkt_close
```

## 2. Train & Test seq_to_seq model (Learner)

<table>
    <thead>
        <tr>
            <th>LC-QuAD2.0</th>
            <th align="center">Accuracy</th>
            <th align="center">BLEU</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Transformer</td>
            <td align="center">89.3</td>
            <td align="center">81.6</td>
        </tr>
        <tr>
            <td>BiLSTM</td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </tbody>
</table>
N.B. Since we encode all entities & relations under the same name, the sparql vocab size is small and the BLEU score is impressively big

## 3. Decode data & Entity linking (Interpreter)

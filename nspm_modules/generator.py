import pandas as pd
from sparql_parser.SPARQL_parser import SPARQL
from sklearn.model_selection import train_test_split
import spacy
import re


class Generator():
    def __init__(self, src_list, trg_list):
        self.sparql_func = ["select", "distinct", "ask", "where","count", "as", "filter", "contains",
                            "order by", "desc", "asc", "limit", "strstart"]

        self.src_data = list(map(self.clean_txt, src_list))
        self.trg_data = list(map(self.encode_query, trg_list))


    def encode_query(self, query):
        """ Encoding sparql queries into a more suited form for the seq_to_seq model

            SELECT ?answer WHERE { wd:Q169794 wdt:P26 ?X . ?X wdt:P22 ?answer}
                       ====>
            select var1 where brack_open wd_qxxx wdt_p26 var2 sep_dot var2 wdt_pxxx var1 brack_close
        """
        query = " ".join([w.upper() if w in self.sparql_func else w for w in query.split(' ')])
        sparql_query = SPARQL(query)
        all_var = sparql_query.all_var
        for i in range(len(all_var)):
            query = query.replace(all_var[i], "var" + str(i + 1))

        query = query.replace("{", "bkt_open ")
        query = query.replace("}", " bkt_close")
        query = query.replace("(", " par_open ")
        query = query.replace(")", " par_close ")
        query = query.replace(" . ", " sep_dot ")
        query = re.sub(":Q[0-9]*","_qxxx", query)
        query = query.replace("  ", " ").lower()

        if query[0] == ' ':
            return query[1:]
        return query

    def clean_txt(self, txt):
        txt = re.sub('[^a-zA-Z0-9 .\']+', '', txt)
        return txt.lower()


    def generate_train_test_files(self, train_filename, test_filename):
        src_train, src_test, trg_train, trg_test = train_test_split(self.src_data, self.trg_data, test_size=0.10)

        # We create the train and test csv files
        train_df = pd.DataFrame({"question": src_train, "sparql_query": trg_train})
        test_df = pd.DataFrame({"question": src_test, "sparql_query": trg_test})

        train_df.to_csv("data/train_test_files/" + train_filename, index=False)
        test_df.to_csv("data/train_test_files/" + test_filename, index=False)






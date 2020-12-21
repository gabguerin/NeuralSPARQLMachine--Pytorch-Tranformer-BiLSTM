from utils import query_prediction
import re


class Interpreter():
    def __init__(self, model, english, sparql, device):
        self.sparql_func = ["select", "distinct", "ask", "where","count", "as", "filter", "contains",
                            "order by", "desc", "asc", "limit", "strstart"]
        self.model = model
        self.english = english
        self.sparql = sparql
        self.device = device

    def decode_query(self, query):
        """
            select var1 where bkt_open wd_q169794 wdt_p26 var2 sep_dot var2 wdt_p22 var1 bkt_close
                       ====>
            SELECT var1 WHERE {wd:Q169794 wdt:P26 var2 . var2 wdt:P22 var1}
        """
        query = " ".join([w.upper() if w in self.sparql_func else w for w in query.split(' ')])
        query = query.replace("bkt_open ", "{")
        query = query.replace(" bkt_close", "}")
        query = query.replace(" prts_open ", '(')
        query = query.replace(" prts_close ", ")")
        query = query.replace("sep_dot", ".")
        query = query.replace("_", ":")
        query = query.replace("pxxx", "PXXX")
        query = query.replace("qxxx", "QXXX")
        return query

    def clean_txt(self, txt):
        txt = re.sub('[^a-zA-Z0-9 .\']+', '', txt)
        return txt.lower()

    def query_from_question(self, question):
        tokens = self.clean_txt(question).split(' ')
        prediction = query_prediction(self.model, tokens, self.english, self.sparql, self.device)[:-1]
        query = " ".join(prediction)
        return self.decode_query(query)
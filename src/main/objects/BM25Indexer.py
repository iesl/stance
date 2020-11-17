import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer


class InstituteCorpus:
    def __init__(self, institutes_db_file, lowercase=True, ngram_range=(1, 1), analyzer="char_wb"):
        """
        analyzer: string, {‘word’, ‘char’, ‘char_wb’} or callable, default=’char_wb’
        """
        self.institute_db = pd.read_csv(institutes_db_file, index_col=None)
        if "composed_title" not in self.institute_db.columns:
            print("Adding composed title...")
            self.institute_db["composed_title"] = self.institute_db.apply(self.compose_university_title, axis=1)
            self.institute_db.to_csv("debug.csv", index=None)
            print("Done")
        self.institute_db = self.institute_db.loc[self.institute_db["TYPE"] != "City"]
        self.institute_db = self.institute_db.loc[self.institute_db["TYPE"] != "State"]
        self.institute_db.reset_index(drop=True, inplace=True)
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.analyzer = analyzer

        self.analyzer = CountVectorizer(lowercase=self.lowercase, ngram_range=self.ngram_range, analyzer=self.analyzer).build_analyzer()
        self.index = None
        self.build_index()

    @staticmethod
    def compose_university_title(df_row):
        return ", ".join(df_row[["NAME", "CITY", "STATE", "ZIPCODE", "COUNTRY"]].fillna("").tolist())

    def build_index(self):
        print("Building corpus index...")
        corpus = self.institute_db["composed_title"].apply(self.analyzer).tolist()
        self.index = BM25Okapi(corpus)
        print("Done")

    def get_institute_by_rowids(self, idx):
        return self.institute_db.loc[idx, ["NAME", "CITY", "STATE", "COUNTRY", "ZIPCODE", "ID", "ID_SOURCE"]]

    def get_institute_title_by_rowids(self, idx):
        return self.institute_db.loc[idx, "composed_title"].tolist()

    def query_institute_corpus(self, query, n=10, return_university_title=False):
        if isinstance(query, str):
            scores = self.index.get_scores(self.analyzer(query))
            top_n = np.argsort(scores)[::-1][:n]
            if return_university_title:
                return self.institute_db.loc[top_n, "composed_title"].tolist()
            else:
                return top_n
        elif isinstance(query, list):
            return [self.query_institute_corpus(single_query, n, return_university_title)
                    for single_query in tqdm(query, desc="Processing queries")]
        else:
            raise ValueError(f"Unhandled query type: {type(query)}")

from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile, datapath
from nltk import word_tokenize 
import os
import pandas as pd

# inference
def inference(model_path, dictionary_path, index_path, query, k, text_path):

    modelss = models.LdaModel.load(datapath(model_path))
    dictionarys = corpora.Dictionary.load_from_text(get_tmpfile(dictionary_path))   
    indexs = similarities.SparseMatrixSimilarity.load(get_tmpfile(index_path))

    query_vector = dictionarys.doc2bow(word_tokenize(query))
    sim = indexs[modelss[query_vector]]

    sim_k = sorted(enumerate(sim), key=lambda item: item[1], reverse=True)[:k]
    
    df = pd.read_csv(text_path, sep=';', header=0, index_col=None)
    reflection = df.loc[sim_k[0][0], "response_bot"]
    
    return sim_k, reflection
    


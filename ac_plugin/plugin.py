from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile, datapath
from nltk import word_tokenize 
import os
import pandas as pd


def get_training_data(text_path):
    df = pd.read_csv(text_path, sep=';', header=0, index_col=None)
    texts = []
    for i in range(len(df)):
        texts.append(str(df.loc[i, 'context_bot']) + ' SEP ' + str(df.loc[i, 'context_user']))
    
    return texts


def train(text_path, model_path, dictionary_path, index_path):

    texts = get_training_data(text_path) 
    texts = [word_tokenize(text) for text in texts]

    dictionary = corpora.Dictionary(texts)
    leng_dict = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]

    model = models.LdaModel(corpus) 
    index = similarities.SparseMatrixSimilarity(model[corpus], num_features=leng_dict)

    model.save(datapath(model_path))
    dictionary.save_as_text(get_tmpfile(dictionary_path))
    index.save(get_tmpfile(index_path))
        
    return model, dictionary, index


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


if __name__=="__main__":
    model_path, dictionary_path, index_path = os.getcwd()+'/save/model', os.getcwd()+'/save/dictionary', os.getcwd()+'/save/index'
    text_path = os.getcwd()+'/reflection.csv'

    # # Training
    # model, dictionary, index = train(text_path, model_path, dictionary_path, index_path)

    # Inference
    query = 'Why did you rate it a 2 but not a 4? What would make you more confident in achieving your step count goal and make you rate your confidence a 4? [SEP] If I can be stronger.'
    sim_k, reflection = inference(model_path, dictionary_path, index_path, query, 3, text_path)

    print(f'query is: {query}\n')
    print(f'Reflection is: {reflection}')
    


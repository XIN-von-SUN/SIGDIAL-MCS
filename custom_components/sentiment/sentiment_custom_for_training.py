"""
## Folowing code is for training sentiment classifier by sklearn
"""

"""
import numpy as np
import pandas as pd
import re, pickle, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
import os
current_path = os.getcwd()

def pre_processing(X):
    documents = []
    # stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        # document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents

def text2vec(documents):
    # vectorizer = CountVectorizer(ngram_range=(2,2), stop_words=stopwords.words('english'))
    vectorizer = CountVectorizer(ngram_range=(2,2))
    X = vectorizer.fit_transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    return X, vectorizer, tfidfconverter

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = svm.SVC().fit(X_train, y_train) 
    
    return classifier

def save_model(classifier, vectorizer, tfidfconverter, model_path, vectorizer_path, tfidfconverter_path):
    with open(model_path, 'wb+') as classifier_file:
        pickle.dump(classifier, classifier_file)
        classifier_file.close()

    with open(vectorizer_path, 'wb+') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
        vectorizer_file.close()

    with open(tfidfconverter_path, 'wb+') as tfidfconverter_file:
        pickle.dump(tfidfconverter, tfidfconverter_file)
        tfidfconverter_file.close()

def load_model(model_path, vectorizer_path, tfidfconverter_path):
    with open(model_path, 'rb') as training_model:
        classifier = pickle.load(training_model)

    with open(vectorizer_path, 'rb') as vectorizer:
        vectorizer = pickle.load(vectorizer)

    with open(tfidfconverter_path, 'rb') as tfidfconverter:
        tfidfconverter = pickle.load(tfidfconverter)

    return classifier, vectorizer, tfidfconverter

def pred(classifier, vectorizer, tfidfconverter, text):
    text = pre_processing([text])
    
    X = vectorizer.transform(text).toarray()
    X = tfidfconverter.transform(X).toarray()
    sent_polarity = classifier.predict(X)
    
    return sent_polarity[0]


if __name__ == '__main__': 
    model_path, vectorizer_path, tfidfconverter_path = current_path+"/custom_component_data/classifier.pickle", current_path+"/custom_component_data/vectorizer.pickle", current_path+"/custom_component_data/tfidfconverter.pickle"        
    
    # Training
    # df = pd.read_table(current_path+"/custom_component_data/sentiment_data.txt", sep=" ")
    # x_raw, y = df.Sentence, df.Label

    # documents = pre_processing(x_raw)
    # X, vectorizer, tfidfconverter = text2vec(documents)
    # classifier = train(X, y)
    # save_model(classifier, vectorizer, tfidfconverter, model_path, vectorizer_path, tfidfconverter_path)

    # Test
    input_text = "not really"
    sid = SentimentIntensityAnalyzer()
    res = sid.polarity_scores(input_text)
    keep = {'neg', 'neu', 'pos'}
    res_vader = {key: value for key, value in res.items() if key in keep}
    key_vader, value = max(res_vader.items(), key=lambda x: x[1])
    print("key_vader: ", key_vader)

    classifier, vectorizer, tfidfconverter = load_model(model_path, vectorizer_path, tfidfconverter_path)
    res_custom = pred(classifier, vectorizer, tfidfconverter, input_text)
    key_custom = "pos" if res_custom == 1 else "neg"
    print("key_custom: ", key_custom)
"""



#########################################################################################################################################################################################



"""
## Folowing code is for training sentiment classifier by Bert
"""
import numpy as np
import pandas as pd
import re, pickle, nltk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AdamW
import os
current_path = os.getcwd()


class OwnData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def data_load(df_own_path, df_twt_path_train, df_twt_path_test):

    df_twt_train = pd.read_csv(df_twt_path_train, sep=",").sample(frac=0.3, replace=False)
    df_twt_test = pd.read_csv(df_twt_path_test, sep=",").sample(frac=0.5, replace=False)
    df_own = pd.read_table(df_own_path, sep=" ")
    x_own_raw, y_own_raw = df_own.text.values, df_own.sentiment.values

    df_twt_train["sentiment"].replace(to_replace="negative", value=0, inplace=True)
    df_twt_train["sentiment"].replace(to_replace="neutral", value=2, inplace=True)
    df_twt_train["sentiment"].replace(to_replace="positive", value=1, inplace=True)
    df_twt_test["sentiment"].replace(to_replace="negative", value=0, inplace=True)
    df_twt_test["sentiment"].replace(to_replace="neutral", value=2, inplace=True)
    df_twt_test["sentiment"].replace(to_replace="positive", value=1, inplace=True)

    x_raw, y_raw = list(df_twt_train.text.values) + (list(x_own_raw)), list(df_twt_train.sentiment.values) + (list(y_own_raw))
    documents = pre_processing(x_raw)

    train_texts, train_labels = documents, y_raw
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    test_texts, test_labels = pre_processing(list(df_twt_test.text.values)), list(df_twt_test.sentiment.values)
    
    print("Training size: ", len(train_labels))
    print("Val size: ", len(val_labels))
    print("Test size: ", len(test_labels))
    
    return train_texts, val_texts, train_labels, val_labels, test_texts, test_labels

def pre_processing(X):
    documents = []
    # stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        # document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents


def train(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, epoch_num, model_path, tokenizer_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Runing on: ", device)
    
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = OwnData(train_encodings, train_labels)
    val_dataset = OwnData(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    # Training part
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epoch_num):
        print("epoch: ", epoch, "Training................")

        train_loss, train_accuracy = [], []

        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # print("label size: ", labels.size())
            # print("label: ", labels)
        
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # print("outputs: ", outputs)
            loss = outputs[0]
            loss.backward()
            optim.step()

            # print("output: ", outputs)
            # print("loss: ", loss)

            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        print("train_loss: ", train_loss)

        # Evaluation part
        model.eval()
        # Tracking variables
        val_accuracy = []
        val_loss = []
        # For each batch in our validation set...
        for batch_val in val_loader:
            # Load batch to GPU
            # b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            input_ids = batch_val['input_ids'].to(device)
            attention_mask = batch_val['attention_mask'].to(device)
            labels = batch_val['labels'].to(device)

            # Compute logits
            with torch.no_grad():
                logits = model(input_ids, attention_mask)

            # Compute loss
            loss = loss_fn(logits[0], labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits[0], dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        print("val_loss: ", val_loss)
        print("val_accuracy: ", val_accuracy)
    
        # Test the model on test set
        test(test_texts, test_labels, tokenizer, model, device)

        print("--------------------------------------------------------")
        print("\n")

    # save tokenizer and model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print('Model and Tokenizer are saved!')
    
    return tokenizer, model, device

def test(test_texts, test_labels, tokenizer, model, device):

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = OwnData(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    model.eval()
    # Tracking variables
    test_accuracy = []
    test_loss = []

    # For each batch in our validation set...
    for batch_test in test_loader:
        # Load batch to GPU
        # b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        input_ids = batch_test['input_ids'].to(device)
        attention_mask = batch_test['attention_mask'].to(device)
        labels = batch_test['labels'].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        # Compute loss
        loss = loss_fn(logits[0], labels)
        test_loss.append(loss.item())
        
        # Get the predictions
        preds = torch.argmax(logits[0], dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    loss = np.mean(test_loss)
    accuracy = np.mean(test_accuracy)

    print("Test_loss: ", loss)
    print("Test_accuracy: ", accuracy)

def pred(model, tokenizer, input_text):
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    # model = BertForSequenceClassification.from_pretrained(model_path)
    
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    #sentiment = "pos" if classifier(input_text)[0]['label']=='LABEL_1' else "neg"
    if classifier(input_text)[0]['label']=='LABEL_0':
        sentiment = "neg" 
    elif classifier(input_text)[0]['label']=='LABEL_1':
        sentiment = "pos"
    else:
        sentiment = "neu" 
    
    score = classifier(input_text)[0]['score']
    
    return sentiment, score

if __name__ == '__main__': 
    df_own_path = current_path + "/sentiment_data_model/sentiment_data/own_sentiment_data.txt"
    df_twt_path_train = current_path + "/sentiment_data_model/sentiment_data/train.csv"
    df_twt_path_test = current_path + "/sentiment_data_model/sentiment_data/test.csv"

    model_path = current_path + "/sentiment_data_model/finetune_sent_bert"
    tokenizer_path = model_path + "/tokenizer/tokenizer_finetune_sent_bert"

    train_texts, val_texts, train_labels, val_labels, test_texts, test_labels = data_load(df_own_path, df_twt_path_train, df_twt_path_test)

    epoch_num = 30
    tokenizer, model, device = train(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, epoch_num, model_path, tokenizer_path)

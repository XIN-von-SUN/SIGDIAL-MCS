"""
## Folowing code is for training sentiment classifier by 
"""
import numpy as np
import pandas as pd
import re, pickle, nltk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
# loss_fn = nn.CrossEntropyLoss()

from torch import cuda
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import random, time, datetime
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, AdamW, BartTokenizer, BartForSequenceClassification, BartModel, BartConfig, T5Tokenizer, T5Model

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

    df_twt_train = pd.read_csv(df_twt_path_train, sep=",").sample(frac=0.75, replace=False)
    df_twt_test = pd.read_csv(df_twt_path_test, sep=",")
    df_own = pd.read_table(df_own_path, sep=" ")
    x_own_raw, y_own_raw = df_own.Sentence.values, df_own.Label.values

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
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = OwnData(train_encodings, train_labels)
    val_dataset = OwnData(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epoch_num):
        print("epoch: ", epoch, "Training................")
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
    # save tokenizer and model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    
    return tokenizer, model, device

def test(test_texts, test_labels, tokenizer, model, device):

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = OwnData(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
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


########################################################################################################################################################################################################
# Test on pre-trained model variants

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def data_loader(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):

    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", do_lower_case=True)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    # print("train_encodings: ", len(train_encodings['input_ids'][50]))
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # print('train_encodings', train_encodings)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # onehot encoding the labels
    train_labels = LabelBinarizer().fit_transform(train_labels)
    val_labels = LabelBinarizer().fit_transform(val_labels)
    test_labels = LabelBinarizer().fit_transform(test_labels)

    train_dataset = OwnData(train_encodings, train_labels)
    val_dataset = OwnData(val_encodings, val_labels)
    # print('train_dataset', train_dataset)
    test_dataset = OwnData(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print('train_loader', train_loader)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # self.l1 = BertModel.from_pretrained("bert-base-cased")
        self.l1 = BartModel.from_pretrained("facebook/bart-large")
        self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 3)
        self.l3 = torch.nn.Linear(1024, 3)
    
    # def forward(self, ids, mask, token_type_ids):
    def forward(self, ids, mask):
        #_, output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        # output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)[0][:, 0, :]
        output_1 = self.l1(ids, attention_mask = mask)[0][:, 0, :]
        #print(output_1.size())
        #print(output_1[0].size())
        #print(output_1)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


if __name__ == '__main__': 
    df_own_path = current_path + "/sentiment_data_model/sentiment_data/sentiment_data.txt"
    df_twt_path_train = current_path + "/sentiment_data_model/sentiment_data/train.csv"
    df_twt_path_test = current_path + "/sentiment_data_model/sentiment_data/test.csv"

    # model_path = current_path + "/sentiment_data_model/finetune_sent_bert"
    # tokenizer_path = model_path + "/tokenizer/tokenizer_finetune_sent_bert"

    train_texts, val_texts, train_labels, val_labels, test_texts, test_labels = data_load(df_own_path, df_twt_path_train, df_twt_path_test)
    train_loader, val_loader, test_loader = data_loader(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)


    model = ModelClass()
    # Tell pytorch to run this model on the GPU.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Runing on: ", device)

    model.to(device)
    # model.cuda()

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_loader) * epochs
    print('total_steps: ', total_steps)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        logits_set = []
        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device, dtype = torch.float)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        # token_type_ids=None, 
                        mask=b_input_mask
                        # labels=b_labels
                        )
            # print('outputs size: ', outputs.size())
            # print('outputs: ', outputs)

            outputs = outputs.float()
            # print('outputs: ', outputs)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.

            # loss, logits = outputs[:2]
            # print(b_labels.type)
            # print('b_labels: ', b_labels)
            # b_labels = b_labels.float()
            # print('b_labels: ', b_labels)

            loss = loss_fn(outputs, b_labels)
        
            # logits_set.append(outputs)
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_loader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(format_time(time.time() - t0)))
            

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        fin_targets=[]
        fin_outputs=[]
        
        # Evaluate data for one epoch
        for batch_val in val_loader:
            
            # Add batch to GPU
            # batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            # b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = batch_val['input_ids'].to(device)
            b_input_mask = batch_val['attention_mask'].to(device)
            b_labels = batch_val['labels'].to(device, dtype = torch.float)
            
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids, 
                                # token_type_ids=None, 
                                mask=b_input_mask
                                # labels=b_labels
                                )
            
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            # loss, logits = outputs[:2]

            outputs = outputs.float()
            # b_labels = b_labels.float()

            loss = loss_fn(outputs, b_labels)
            # print('eval loss is: ', loss)

            # Move logits and labels to CPU
            # logits = outputs.detach().cpu().numpy()
            # #print('logits is: ', logits)

            # label_ids = b_labels.to('cpu').numpy()
            #print('label is: ', label_ids)

            fin_targets.extend(b_labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            # # Calculate the accuracy for this batch of test sentences.
            # tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # # Accumulate the total accuracy.
            # eval_accuracy += tmp_eval_accuracy

            # # Track the number of batches
            # nb_eval_steps += 1

        print('eval loss is: ', loss)

        fin_targets = np.argmax(fin_targets, axis=1)
        fin_outputs = np.argmax(fin_outputs, axis=1)

        # Report the final accuracy for this validation run.
        print("Accuracy: {0:.2f}".format(metrics.accuracy_score(fin_targets, fin_outputs)))
        print("Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")         
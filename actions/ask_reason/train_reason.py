import pandas as pd 
import os
import yaml
from datasets import Dataset
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_yaml(load_path):
    responses_file = open(load_path, 'r')
    responses = responses_file.read()
    responses = yaml.safe_load(responses)
    return responses

def data_reformulate(df_raw):
    reason, text, label = [], [], []
    for i in df_raw['reasons']:
        example = i['examples'].strip().split('\n')
        for rea in example:
            reason.append(i['reason'])
            text.append(rea)
            label.append(i['label'])
    df_reason = pd.DataFrame({'reason':reason, 'label':label, 'text':text})
    return df_reason


def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(batch['text'], padding='max_length', max_length=30, truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }   


def model_init():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11)


def run_hyper_param_search(load_data_path, save_model_path, batch_size, num_search_trials):
    df_raw = load_yaml(load_data_path)
    df_reason = data_reformulate(df_raw)
    dataset = Dataset.from_pandas(df_reason)

    dataset = dataset.shuffle()
    train_dataset = dataset.train_test_split(test_size=0.1)['train']
    test_dataset = dataset.train_test_split(test_size=0.1)['test']

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=batch_size)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
                output_dir='./training_results',
                # learning_rate=1e-4,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                # num_train_epochs=10,
                weight_decay=0.005,
                logging_dir='./logs'
            )

    trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

    best_run = trainer.hyperparameter_search(n_trials=num_search_trials, direction="maximize")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()
    trainer.save_model(save_model_path)
    print(f'evaluate res is: {trainer.evaluate()}')


def run_train(load_data_path, save_model_path, batch_size):
    df_raw = load_yaml(load_data_path)
    df_reason = data_reformulate(df_raw)
    dataset = Dataset.from_pandas(df_reason)

    dataset = dataset.shuffle()
    train_dataset = dataset.train_test_split(test_size=0.1)['train']
    test_dataset = dataset.train_test_split(test_size=0.1)['test']

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=batch_size)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
                output_dir='./training_results',
                learning_rate=1e-4,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=30,
                weight_decay=0.005,
                # logging_dir='./logs'
            )

    trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

    trainer.train()
    trainer.save_model(save_model_path)
    print(f'evaluate res is: {trainer.evaluate()}')


if __name__=="__main__":
    current_path = os.getcwd()
    load_data_path, save_model_path = 'reasons.yml', current_path+'/model'
    batch_size = 8

    run_train(load_data_path, save_model_path, batch_size)

    # num_search_trials = 10
    # run_hyper_param_search(load_data_path, save_model_path, batch_size, num_search_trials)
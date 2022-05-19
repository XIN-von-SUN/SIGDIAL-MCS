from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AdamW
import sentiment_custom_for_training as sent
import os
current_path = os.getcwd()

if __name__ == '__main__': 

    model_path = current_path + "/sentiment_data_model/sentiment_data/finetune_sent_bert"
    tokenizer_path = model_path + "/tokenizer/tokenizer_finetune_sent_bert"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    input_texts = ['today is monday', 'i am not sure', 'i dont think so', 'do not think', 'dont know', 'yes obese', 'i am not obese', 'not really', 'why not', 'no i am not obese', 'what a sad day', 'good day']
    
    for i in input_texts:
        sentiment, score = sent.pred(model, tokenizer, i)
        print(i)
        print(sentiment, score)

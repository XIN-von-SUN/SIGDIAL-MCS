import os
from transformers import pipeline, BertTokenizer, BertForSequenceClassification


def predict(save_model_path, input_text):
    model = BertForSequenceClassification.from_pretrained(save_model_path)
    tokenizer = BertTokenizer.from_pretrained(save_model_path)

    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    pred_label = classifier(input_text)[0]['label']

    idx2label = {"LABEL_0": "control_weight",
                "LABEL_1": "live_healthy",
                "LABEL_2": "against_diseases",
                "LABEL_3": "decrease_risk_heart_attack",
                "LABEL_4": "reduce_stress",
                "LABEL_5": "reduce_depression",
                "LABEL_6": "better_sleep",
                "LABEL_7": "social",
                "LABEL_8": "gym_body_fit",
                "LABEL_9": "more_risk_heart_attack",
                "LABEL_10": "waste_time"
            }

    # print(f'predicted label is: {pred_label}, {idx2label[pred_label]}')
    
    return idx2label[pred_label]
    

if __name__=="__main__":
    current_path = os.getcwd()
    save_model_path = current_path+'/model'

    input_text = input('Pls say something: ')
    pred_label = predict(save_model_path, input_text)
    print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'exercise can give me better sleep'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'physical activity can help me lose weight'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'i want to be stronger'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'exercise can take depression from me'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'make friends when doing exercise'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'exercise is healthy lifestyle'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'increase the risk of heart attack'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')

    # input_text = 'waste of my time'
    # pred_label = predict(save_model_path, input_text)
    # print(f'text is: {input_text}, pred_label is: {pred_label}')




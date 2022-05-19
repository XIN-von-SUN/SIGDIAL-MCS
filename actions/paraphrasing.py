"""
from transformers import AutoTokenizer, AutoModelWithLMHead 

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("salesken/paraphrase_generation")  
model = AutoModelWithLMHead.from_pretrained("salesken/paraphrase_generation").to(device)

input_query="i feel so sad today because I lost my wallet."
query = input_query + " ~~ "

input_ids = tokenizer.encode(query.lower(), return_tensors='pt').to(device)
sample_outputs = model.generate(input_ids,
                                do_sample=True,
                                num_beams=1, 
                                max_length=128,
                                temperature=0.9,
                                top_p= 0.99,
                                top_k = 30,
                                num_return_sequences=40)
paraphrases = []
for i in range(len(sample_outputs)):
    r = tokenizer.decode(sample_outputs[i], skip_special_tokens=True).split('||')[0]
    r = r.split(' ~~ ')[1]
    if r not in paraphrases:
        paraphrases.append(r)

print(paraphrases)
"""


"""
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("salesken/natural_rephrase")
model = AutoModelWithLMHead.from_pretrained("salesken/natural_rephrase")


Input_query = "i feel so sad today because I lost my wallet." #"Hey Siri, Send message to mom to say thank you for the delicious dinner yesterday"
query= Input_query + " ~~ "
input_ids = tokenizer.encode(query.lower(), return_tensors='pt')
sample_outputs = model.generate(input_ids,
                            do_sample=True,
                            num_beams=1, 
                            max_length=len(Input_query),
                            temperature=0.2,
                            top_k = 10,
                            num_return_sequences=1)
for i in range(len(sample_outputs)):
    result = tokenizer.decode(sample_outputs[i], skip_special_tokens=True).split('||')[0].split('~~')[1]
    print(result)
"""



"""
import numpy as np
import random
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class ReflectiveListening:
    """"""
    A class to generate reflective listening statements via paraphrase generation
    For example:
    Statement: "My teeth can be sensitive at times due to TMJ issues."
    Reflective listening response: "I understand, so your teeth are sensitive due to temporomandibular disorders."
    """"""
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self.model_name = 'tuner007/pegasus_paraphrase'
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(self.model_name) \
            .to(ReflectiveListening.torch_device)

    def get_paraphrase(self, raw_text):
        """"""
        Obtains paraphrase of a text using the PEGASUS model https://huggingface.co/tuner007/pegasus_paraphrase
        20 candidate paraphrases are generated using beam search, and scored against the Parametric score. The highest
        scoring paraphrase is returned.
        :param raw_text: Original text to be paraphrased
        :return: Paraphrase with the highest Parametric score
        """"""
        batch = self.pegasus_tokenizer([raw_text], truncation=True, padding='longest', max_length=60,
                                       return_tensors="pt").to(ReflectiveListening.torch_device)
        paraphrased = self.pegasus_model.generate(
            **batch,
            max_length=60,
            num_beams=10,
            num_return_sequences=5,
            temperature=1.5,
            early_stopping=True
        )
        paraphrases = self.pegasus_tokenizer.batch_decode(paraphrased, skip_special_tokens=True)
        
        return paraphrases

    def get_response(self, raw_text):
        """"""
        Obtains the final response by paraphrasing the input text, then flipping the P.O.V, and concatenating standard
        reflective listening phrases at the start e.g. "I understand, "
        """"""
        paraphrase = random.choice(self.get_paraphrase(raw_text))
        flipped = flip_pov(paraphrase)
        response = concat_start(flipped)

        return response # paraphrase


if __name__=="__main__":
    inputs = "I feel so sad today becasue I lost my bag."
    reflection = ReflectiveListening()
    response = reflection.get_response(inputs)
    print("response is: ", response)
"""


import random
import requests

def concat_start(text):
    """Concatenate standard reflective listening phrases"""
    starts = ["It sounds like ", "I understand, seems like ", "I get a sense that ", "It seems like ", "I see, so "]
    return random.choice(starts) + text


def flip_pov(text):
    """Flip the P.O.V from the speaker to the listener (I <-> you)"""
    subject_flip = {
        "I": "you",
        "my": "your",
        "My": "Your",
        "I'm": "you're",
        "am": "are",
        "we": "you",
        "We": "You",
        "me": "you",
        "myself": "yourself",
        "Myself": "Yourself",
        "I'd": "you'd",
    }
    text = text.strip('.').split()
    for idx, word in enumerate(text):
        if word in subject_flip:
            text[idx] = subject_flip[word]
    text = ' '.join(text)
    lowercase = lambda s: s[:1].lower() + s[1:] if s else ''
    return lowercase(text)


"""
API call
"""
API_TOKEN = "api_RvFdtNqolCvyPJiITvdjojFsrfGHuZGGNm"
API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_paraphrase"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    # print(f'response is: {response.json()}')
    return response.json()


def get_response(raw_text):
    """
    Obtains the final response by paraphrasing the input text, then flipping the P.O.V, and concatenating standard
    reflective listening phrases at the start e.g. "I understand, "
    """
    flipped = flip_pov(raw_text)
    response = concat_start(flipped)
    return response


def segment(input_all):
    seg = [".", "..", "...", "?", "!"]
    res, words = [], []
    for chr in input_all:
      if chr in seg:
        if len(words) != 0:
            sent = ''.join(words)
            sent += '.'
            res.append(sent)
        words = []
      else:
        words.append(chr)
    sent = ''.join(words)
    sent += '.'
    res.append(sent)

    for i in res:
        if i=='.':
            res.remove(i)
    return res


def reflection(input_all):
    sentences = segment(input_all)
    responses = []
    # print('1')
    for sent in sentences:
        # print('2.1')
        # print(f'sent is: {sent}')
        output = query({"inputs": sent})[0]["generated_text"]
        responses.append(output)
        # print(f'output is: {output}')
        # print('2.2')
    responses = ' '.join(responses)
    # print('3')
    reflection = get_response(responses)
    # print('4')
    return reflection


if __name__=="__main__":
    input_all =  "I want to lose weight, but I just love to eat sugary treats!I can not stop myself when they are in the house? I know it is not good for me." 
    # output = query({"inputs": input_all})[0]["generated_text"]
    # res = get_response(output)
    # print(res)
    reflection = reflection(input_all)
    print(f'reflection is: {reflection}')
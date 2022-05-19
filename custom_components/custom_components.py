from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re, pickle
import custom_components.sentiment.sentiment_custom_for_training as sent
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# from googletrans import Translator

current_path = os.getcwd()

# """
# Following is the sentiment component by pure Bert
# """
# class SentimentAnalyzer(Component):
#     """
#     A pre-trained sentiment component + Own custom sentiment component
#     """
#     name = "sentiment"
#     provides = ["sentiment"]
#     requires = []
#     defaults = {}
#     language_list = ["en"]

#     def __init__(self, component_config=None):
#         super(SentimentAnalyzer, self).__init__(component_config)    
#         self.current_path = os.getcwd() + "/custom_component/sentiment/sentiment_data_model"
#         self.model_path, self.tokenizer_path = self.current_path+"/finetune_sent_bert", self.current_path+"/finetune_sent_bert/tokenizer/tokenizer_finetune_sent_bert"
#         self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=True)
#         self.model = BertForSequenceClassification.from_pretrained(self.model_path)
    
#     def train(self, training_data, cfg, **kwargs):
#         """Not needed, because the the model is pretrained"""
#         pass

#     def convert_to_rasa(self, value, confidence):
#         """Convert model output into the Rasa NLU compatible output format."""

#         entity = {"value": value,
#                   "confidence": confidence,
#                   "entity": "sentiment",
#                   "extractor": "sentiment_extractor"}
#         return entity

#     def process(self, message: Message, **kwargs):
#         """Retrieve the text message, pass it to the classifier
#             and append the prediction results to the message class."""
#         try:
#             sentiment, score = sent.pred(self.model, self.tokenizer, message.data['text'])
#             entity = self.convert_to_rasa(sentiment, score)
#             message.set("entities", [entity], add_to_output=True)
#         except KeyError:
#             pass
    
#     def persist(self, file_name, model_dir):
#         """Pass because a pre-trained model is already persisted"""
#         pass

"""
Following is the sentiment component by pure Bert
"""
class Analyzer(Component):
    """
    Own custom components
    """
    name = "Analyzer"
    provides = ["Analyzer"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        super(Analyzer, self).__init__(component_config)    
        self.current_path = os.getcwd() + "/custom_components/sentiment/sentiment_data_model"
        self.model_path, self.tokenizer_path = self.current_path+"/finetune_sent_bert", self.current_path+"/finetune_sent_bert/tokenizer/tokenizer_finetune_sent_bert"
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
    
    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass

    def convert_to_rasa(self, value, confidence, component_name):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": component_name,
                  "extractor": "component_extractor"}
        return entity

    def process(self, message: Message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        try:
            sentiment, score = sent.pred(self.model, self.tokenizer, message.data['text'])
            entity_sentiment = self.convert_to_rasa(sentiment, score, "sentiment")

        #     translator = Translator()
        #     res = translator.translate(message.data['text'])
        #     if res.src == "nl":
        #         lang_id = "nl"
        #     elif res.src == "de":
        #         lang_id = "de"
        #     elif res.src == "es":
        #         lang_id = "es"
        #     else:
        #         lang_id = "en"
        #     entity_language = self.convert_to_rasa(lang_id, float(1), "language_id")
        #     message.set("entities", [entity_sentiment, entity_language], add_to_output=True)
        
            message.set("entities", [entity_sentiment], add_to_output=True)
        except KeyError:
            pass
    
    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass



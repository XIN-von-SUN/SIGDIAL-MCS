from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re, pickle
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

current_path = os.getcwd()

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
        pass

    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass

    def convert_to_rasa(self, value, confidence, component_name):
        """Convert model output into the Rasa NLU compatible output format."""

        pass

    def process(self, message: Message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""

        pass
    
    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass



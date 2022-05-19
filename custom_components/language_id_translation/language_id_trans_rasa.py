from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from googletrans import Translator
import os

"""
Following is the language identification and translation component 
"""

class LanguageIdentifier(Component):
    """
    A custom language identification and translation component
    """
    name = "languageId"
    provides = ["languageId"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config=None):
        super(LanguageIdentifier, self).__init__(component_config)    
        # self.current_path = os.getcwd() + "/custom_component/language_id_translation"
    
    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "language_id",
                  "extractor": "identification_extractor"}
        return entity

    def process(self, message: Message, **kwargs):
        """Retrieve the text message, pass it to the identifier
            and append the prediction results to the message class."""
        try:
            translator = Translator()
            res = translator.translate(message.data['text'])
            if res.src == "nl":
                lang_id = "nl"
            elif res.src == "de":
                lang_id = "de"
            elif res.src == "es":
                lang_id = "es"
            else:
                lang_id = "en"
            entity = self.convert_to_rasa(lang_id, float(1))
            message.set("language_id", [entity], add_to_output=True)
        except KeyError:
            pass
    
    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass


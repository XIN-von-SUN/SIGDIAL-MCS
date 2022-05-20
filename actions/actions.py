# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import SlotSet, FollowupAction, ActionExecuted

from dotenv import load_dotenv
import os
import requests
import pandas as pd
import json
import uuid
import random

print(f'current path: {os.getcwd()}')


##################################################################################################


# This module will save all events after the most recent restart
from actions.tracker_read import tracker_read
record_file = 'data/dialogue_history/events_after_latest_restart.json'
pure_dialog_records = 'data/dialogue_history/pure_dialog_records.txt'

class RecordHistory(Action):

    def name(self) -> Text:
        return "action_record_history"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        events_after_latest_restart = tracker.events_after_latest_restart()
        with open(record_file, 'w+') as file:
            #for event in events_after_latest_restart:
            events_json = json.dumps(events_after_latest_restart)
            file.write(events_json)
            file.close()

        tracker_read(record_file, pure_dialog_records)
        dispatcher.utter_message("All our dialogue history has been saved!")

        return []


##################################################################################################


# import actions.paraphrasing as parap
import ac_plugin.plugin as plugin
from nltk.tokenize import sent_tokenize
ac_plugin_path = os.getcwd() + '/ac_plugin'

def nlg_plugin(tracker, dispatcher):
    last_bot_event = next(e for e in reversed(tracker.events) if e["event"]=="bot")["text"]
    user_input_text = tracker.latest_message["text"]

    model_path, dictionary_path, index_path = ac_plugin_path+'/save/model', ac_plugin_path+'/save/dictionary', ac_plugin_path+'/save/index'
    text_path = ac_plugin_path+'/reflection.csv'

    query = last_bot_event + ' \n ' + user_input_text
    sim_k, reflection = plugin.inference(model_path, dictionary_path, index_path, query, 2, text_path)

    print(f'reflection: {str(reflection)}\n')

    if reflection != 'nan':
        reflection_seg_sent = sent_tokenize(reflection)
        for reflection_sent in reflection_seg_sent:
            dispatcher.utter_message(text=reflection_sent)
    else:
        pass

    return reflection

    
##################################################################################################


import ac_connector.connector as connector
ac_connector_path = os.getcwd() + '/ac_connector'

class ConnectorAskNextTopic(Action):

    def name(self) -> Text:
        return "action_connector_change_topic_model"

    async def run(self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

            all_topic = tracker.get_slot("all_topic")
            if all_topic == None:
                all_topic = ["switch_rating_importance", "switch_pa"]

            switch_pa = tracker.get_slot("switch_pa").lower() if tracker.get_slot("switch_pa") else None
            switch_rating_importance = tracker.get_slot("switch_rating_importance").lower() if tracker.get_slot("switch_rating_importance") else None
            switch_greeting = tracker.get_slot("switch_greeting").lower() if tracker.get_slot("switch_greeting") else None

            topic_dict = {
                        "switch_rating_importance": switch_rating_importance, 
                        "switch_pa": switch_pa,
                        "switch_greeting": switch_greeting,
                        }

            for i in all_topic:
                if topic_dict[i] != None:
                    all_topic.remove(i)

            if all_topic == []:
                utter_no_topic_left = f"It seems we have talked a lot! I will catch you up next time! See you soon!"
                dispatcher.utter_message(text=utter_no_topic_left)
                return [SlotSet("all_topic", all_topic), SlotSet("next_topic", None)]
            else:
                next_topic = random.choice(all_topic)
                print(f'next topic random: {next_topic}\n')
                utter_dict = {
                            "switch_self_efficacy": "utter_ask_permission_topic_self_efficacy", 
                            "switch_pa": "utter_ask_permission_topic_pa",
                            "switch_rating_importance": "utter_ask_permission_topic_rating_importance", 
                            }
                all_topic.remove(next_topic)
                utter_stop_current_topic = f"Seems we can move on to the next topic!"
                dispatcher.utter_message(text=utter_stop_current_topic)
                # dispatcher.utter_message(response=utter_dict[next_topic])
                utter_ask_permission_response = utter_dict[next_topic]
                return [SlotSet("all_topic", all_topic), SlotSet("next_topic", next_topic), SlotSet(next_topic, next_topic), FollowupAction(name=utter_ask_permission_response)]


##################################################################################################


# This custom action is for rating_importance_form
class AskRatingImportance(Action):

    def name(self) -> Text:
        return "action_ask_importance_rate"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        buttons = [
            {"payload": '0', "title": "0"},
            {"payload": '1', "title": "1"},
            {"payload": '2', "title": "2"},
            {"payload": '3', "title": "3"},
            {"payload": '4', "title": "4"},
            {"payload": '5', "title": "5"},
            {"payload": '6', "title": "6"},
            {"payload": '7', "title": "7"},
            {"payload": '8', "title": "8"},
            {"payload": '9', "title": "9"},
            {"payload": '10', "title": "10"}]

        dispatcher.utter_message(text=f'Great! On a scale from 0 to 10, where 0 is not at all important and 10 is extremely important, where would you say you are?',
                                buttons=buttons)      
        return []


class AskRatingImportanceMoreReason(Action):

    def name(self) -> Text:
        return "action_ask_rating_importance_more_reason"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        importance_rate = tracker.get_slot("importance_rate")
        importance_rate_more = int(importance_rate) + 2
        
        response2 = f"Why did you rate it a {importance_rate} and not a {importance_rate_more}?"
        response3 = f"What would it take to rate it a {importance_rate_more} and make PA  more important to you?"
        
        dispatcher.utter_message(text=response2)
        dispatcher.utter_message(text=response3)

        return []


class RatingImportanceStop(Action):

    def name(self) -> Text:
        return "action_rating_importance_stop"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        rating_importance_more_reason = tracker.get_slot("rating_importance_more_reason")
        importance_rate = tracker.get_slot("importance_rate")
        print(f'rating_importance_more_reason is: {rating_importance_more_reason}')

        reflection = nlg_plugin(tracker, dispatcher) 

        response2 = f"Now I understand what can make you rate more, because you said: {rating_importance_more_reason}."
        dispatcher.utter_message(text=response2)
        
        response3 = f"The remaining questions of this 'importance about physical activity' pipeline are omitted here!"
        dispatcher.utter_message(text=response3)
        
        return []


##################################################################################################


# # This custom action is for pa_satisfy_form
class AskPASatisfactoryThing(Action):

    def name(self) -> Text:
        return "action_ask_pa_satisfactory_thing"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(response="utter_ask_pa_satisfactory_thing")
  
        return []


class AskPASatisfactoryImprove(Action):

    def name(self) -> Text:
        return "action_ask_pa_satisfactory_improve"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        reflection = nlg_plugin(tracker, dispatcher)

        dispatcher.utter_message(response="utter_ask_pa_satisfactory_improve")
  
        return []


# # This custom action is for pa_not_satisfy_form
class AskPANotSatisfactoryThing(Action):

    def name(self) -> Text:
        return "action_ask_pa_not_satisfactory_thing"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(response="utter_ask_pa_not_satisfactory_thing")
  
        return []


class AskPANotSatisfactoryToSatisfyThing(Action):

    def name(self) -> Text:
        return "action_ask_pa_not_satisfactory_to_satisfy_thing"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        reflection = nlg_plugin(tracker, dispatcher)
        dispatcher.utter_message(response="utter_ask_pa_not_satisfactory_to_satisfy_thing")
  
        return []


# # This custom action is for reflection when complete PA pipeline 
class PACompleteReflection(Action):

    def name(self) -> Text:
        return "action_pa_complete_reflection"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        reflection = nlg_plugin(tracker, dispatcher)
  
        return []


##################################################################################################
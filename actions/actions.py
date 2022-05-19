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


# This module will submit the surveyed profile information as slots into backend database
load_dotenv()
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
base_id = os.getenv("BASE_ID")
table_name = os.getenv("TABLE_NAME")


def create_profile_record(base_id, table_name, airtable_api_key, name, age, exercise, gender, heart_issue, obese):
    request_url = f"https://api.airtable.com/v0/{base_id}/{table_name}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {airtable_api_key}",
    }

    data = {
        "fields": {
            "Id": str(uuid.uuid4()),
            "name": name,
            "age": age,
            "exercise": exercise,
            "gender": gender,
            "heart_issue": heart_issue,
            "obese": obese,
        }
    }
    try:
        response = requests.post(
            request_url, headers=headers, data=json.dumps(data)
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    print(f"Response status code: {response.status_code}")
    return response


class SubmitProfileForm(Action):

    def name(self) -> Text:
        return "action_submit_profile_form_to_DB"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name").lower()
        age = tracker.get_slot("age").lower()
        exercise = tracker.get_slot("exercise").lower()
        gender = tracker.get_slot("gender").lower()
        heart_issue = tracker.get_slot("heart_issue").lower()
        obese = tracker.get_slot("obese").lower()

        response = create_profile_record(base_id, table_name, airtable_api_key, name, age, exercise, gender, heart_issue, obese)

        dispatcher.utter_message("Well done! Your profile information have been recorded!")

        return []


##################################################################################################


# This module will fill the required slots value from backend database based on different patients' name
def query_profile_record(base_id, table_name, airtable_api_key, name):
    request_url = f"https://api.airtable.com/v0/{base_id}/{table_name}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {airtable_api_key}",
    }

    try:
        airtable_response = pd.DataFrame(requests.get(
                        request_url, headers=headers,
                        ).json()['records'])['fields']
        record_all = []
        columns = sorted(airtable_response[0].keys())
        for record in airtable_response:
            record_all.append(dict(sorted(record.items())))
        df = pd.DataFrame(record_all, columns=columns)
        query_record = json.loads(df[df.name==name].to_json(orient="records"))[0]

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    return query_record


class FillSlotFromDatabase(Action):

    def name(self) -> Text:
        return "action_fill_slots_from_db"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
    
        name = tracker.get_slot("name").lower()
        query_record = query_profile_record(base_id, table_name, airtable_api_key, name)
        
        slots_all = list(query_record.keys())
        slots_all.remove('Id')

        for feature in slots_all:
            if query_record[feature] is None:
                slots_all.remove(feature)

        return [SlotSet(feature, query_record[feature] if query_record[feature] is not None else None) for feature in slots_all]


class CollectMissProfile(Action):
    """
    This class is used for keeping collecting the profile features if there are some 
    features are missing in the backend database.
    """
    def name(self) -> Text:
        return "action_confirm_profile_after_db"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
    
        name = tracker.get_slot("name").lower()
        query_record = query_profile_record(base_id, table_name, airtable_api_key, name)
        
        slots_all = list(query_record.keys())
        slots_all.remove('Id')
        slots_left = slots_all.copy()

        utter_slots_from_db = 'Here is your stored profile information, please check!'
        for feature in slots_all:
            if query_record[feature] is not None:
                slots_left.remove(feature)
                utter_slots_from_db += ('\n- ' + str(feature) + ': ' + str(query_record[feature]))
        if slots_left:
            dispatcher.utter_message(text=utter_slots_from_db)
            dispatcher.utter_message(response="utter_profile_ask_permission_for_left_slots_survey")
            return []
        else:
            dispatcher.utter_message(text=utter_slots_from_db)
            return []


##################################################################################################


# This module will reset the polarity status
class PolarityReset(Action):

    def name(self) -> Text:
        return "action_polarity_reset"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        return [SlotSet("polarity", None)]


# This module will return the polarity status based on the result of sentiment module
class Polarity(Action):

    def name(self) -> Text:
        return "action_polarity"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # latest_message = tracker.latest_message["text"]
        sentiment = next(tracker.get_latest_entity_values("sentiment"), None) 
        
        # if sentiment == "pos":
        #     return [SlotSet("polarity", "pos")]
        # else:
        #     return [SlotSet("polarity", "neg")]
        return [SlotSet("polarity", sentiment)]


# This module will return the profile status
class Profile(Action):

    def name(self) -> Text:
        return "action_profile"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        sentiment = next(tracker.get_latest_entity_values("sentiment"), None) 
        
        # if sentiment != "neg":
        #     return [SlotSet("profile", "pos")]
        # else:
        #     return [SlotSet("profile", "neg")]   
        return [SlotSet("profile", sentiment)]


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

    if reflection is True and reflection != 'nan':
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

    # async def run(self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    #     all_topic = tracker.get_slot("all_topic")
    #     if all_topic == None:
    #         all_topic = ["switch_rating_importance", "switch_rating_confidence", "switch_self_efficacy"] # ["switch_motivator", "switch_rating_importance"", "switch_pa"]

    #     # switch_motivator = tracker.get_slot("switch_motivator").lower() if tracker.get_slot("switch_motivator") else None
    #     switch_self_efficacy = tracker.get_slot("switch_self_efficacy").lower() if tracker.get_slot("switch_self_efficacy") else None
    #     # switch_pa = tracker.get_slot("switch_pa").lower() if tracker.get_slot("switch_pa") else None
    #     switch_rating_importance = tracker.get_slot("switch_rating_importance").lower() if tracker.get_slot("switch_rating_importance") else None
    #     switch_rating_confidence = tracker.get_slot("switch_rating_confidence").lower() if tracker.get_slot("switch_rating_confidence") else None

    #     topic_dict = {
    #                 "switch_rating_importance": switch_rating_importance, 
    #                 "switch_rating_confidence": switch_rating_confidence,
    #                 "switch_self_efficacy": switch_rating_confidence,
    #                 }

    #     for i in all_topic:
    #         if topic_dict[i] != None:
    #             all_topic.remove(i)

    #     if all_topic == []:
    #         utter_no_topic_left = f"It seems we have talked a lot! I will catch you up next time! See you soon!"
    #         dispatcher.utter_message(response="utter_no_topic_left")
    #         return [SlotSet("all_topic", all_topic), SlotSet("next_topic", None)]
    #     else:
    #         next_topic = random.choice(all_topic)
    #         utter_dict = {
    #                     # "switch_motivator": "utter_ask_permission_topic_motivator", 
    #                     "switch_self_efficacy": "utter_ask_permission_topic_self_efficacy", 
    #                     # "switch_pa": "utter_ask_permission_topic_pa",
    #                     "switch_rating_importance": "utter_ask_permission_topic_rating_importance", 
    #                     "switch_rating_confidence": "utter_ask_permission_topic_rating_confidence"
    #                     }
    #         all_topic.remove(next_topic)
    #         utter_stop_current_topic = f"Seems we can move on to the next topic!"
    #         dispatcher.utter_message(response="utter_stop_current_topic")
    #         dispatcher.utter_message(response=utter_dict[next_topic])
    #         return [SlotSet("all_topic", all_topic), SlotSet("next_topic", next_topic), SlotSet(next_topic, next_topic), FollowupAction(name='action_session_start')]

    async def run(
                self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
        ) -> List[Dict[Text, Any]]:

        left_topic = tracker.get_slot("left_topic") if tracker.get_slot("left_topic") else ["switch_greeting", "switch_pa", "switch_rating_importance", "switch_rating_confidence", "switch_self_efficacy"]
        talked_topic = tracker.get_slot("talked_topic") if tracker.get_slot("talked_topic") else ["switch_greeting", "switch_pa"]
        # print(f'left_topic 1 is: {left_topic}')
        # print(f'talked_topic 1 is: {talked_topic}\n')    

        # dict_switch = {
        #             "switch_greeting": tracker.get_slot("switch_greeting").lower() if tracker.get_slot("switch_greeting") else None,
        #             "switch_pa": tracker.get_slot("switch_pa").lower() if tracker.get_slot("switch_pa") else None,
        #             "switch_rating_importance": tracker.get_slot("switch_rating_importance").lower() if tracker.get_slot("switch_rating_importance") else None, 
        #             "switch_rating_confidence": tracker.get_slot("switch_rating_confidence").lower() if tracker.get_slot("switch_rating_confidence") else None,
        #             "switch_self_efficacy": tracker.get_slot("switch_self_efficacy").lower() if tracker.get_slot("switch_self_efficacy") else None,
        #             }
        # for i in left_topic:
        #     if dict_switch[i] != None:
        #         left_topic.remove(i)

        if left_topic == []:
            utter_no_topic_left = f"It seems we have talked a lot! I will catch you up next time! See you soon!"
            dispatcher.utter_message(response="utter_no_topic_left")
            return [SlotSet("left_topic", left_topic), SlotSet("next_topic", None)]

        else:
            # following 'talked_topic' has already added 'next_topic' in the 'connector_run' function
            talked_topic, utter_ask_permission_response, next_topic = connector.connector_run(ac_connector_path, talked_topic, out_len=6) 
            # print(f'utter_ask_permission is: {utter_ask_permission_response}')
            print(f'next_topic is: {next_topic}')
            # print(f'left_topic is: {left_topic}\ntalked_topic is: {talked_topic}\n')

            if next_topic != None and next_topic in left_topic: # and next_topic not in not_available_pipelines:
                # print('1')
                left_topic.remove(next_topic)
                # talked_topic.append(next_topic)
                # print(f'talked_topic is: {talked_topic}')
    
                utter_stop_current_topic = f"Seems we can move on to the next topic!"
                dispatcher.utter_message(text=utter_stop_current_topic)
                # dispatcher.utter_message(response=utter_ask_permission_response)
                # return [SlotSet("left_topic", left_topic), SlotSet("talked_topic", talked_topic), SlotSet("next_topic", next_topic), SlotSet(next_topic, True), FollowupAction(name='action_session_start')]    
                return [SlotSet("left_topic", left_topic), SlotSet("talked_topic", talked_topic), SlotSet("next_topic", next_topic), SlotSet(next_topic, True), FollowupAction(name=utter_ask_permission_response)]                    
            else: 
                # print('2')
                utter_no_topic_left = f"It seems we have talked a lot! I will catch you up next time! See you soon!"
                dispatcher.utter_message(text=utter_no_topic_left)
                return [SlotSet("left_topic", left_topic), SlotSet("next_topic", None)]


##################################################################################################


# This custom action is for automatic pro-con of PA classification and further talk about pros-cons
import actions.ask_reason.prediction as prediction
import os

class PA_ReasonNLU(Action):

    def name(self) -> Text:
        return "action_pa_reason_nlu"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        input_text = tracker.latest_message["text"]
        
        current_path = os.getcwd()
        save_model_path = current_path+'/ask_reason/model'
        pred_label = prediction.predict(save_model_path, input_text)

        label2response = {"control_weight": "exercise can help you control your weight",
                "live_healthy": "exercise can make you live healthier",
                "against_diseases": "exercise can help you against the disease",
                "decrease_risk_heart_attack": "exercise can decrease the risk of heart attack",
                "reduce_stress": "exercise can reduce your stress",
                "reduce_depression": "exercise can help you defeat the depression",
                "better_sleep": "you can have better sleep after exercise",
                "social": "exercise can help you make more friends",
                "gym_body_fit": "exercise can give you stronger body and muscle",
                "more_risk_heart_attack": "exercise might increase the risk of heart attack for some people have heart diseases",
                "waste_time": "exercise will waste some of the time",
            }
        
        if pred_label in ["more_risk_heart_attack", "waste_time"]:
            response_con_list = [(f'Let me see if I understand, you think the possible cons of people doing exercise and physical activity is '),
                            (f'I get it, you are afraid that ')]
            response_con = random.choice(response_con_list) + str(label2response[pred_label])
            dispatcher.utter_message(text=response_con)
            
            if tracker.get_slot("pa_con") is None:
                return [SlotSet("pa_con", input_text)]
            else:
                pa_con_raw = tracker.get_slot("pa_con")
                pa_con = pa_con_raw + ' /sep ' + input_text
                return [SlotSet("pa_con", pa_con)]
        else:
            response_pro_list = [(f'If I am understand correctly, you think the pros of people doing exercise and physical activity is '),
                            (f'I get it, pyhsical activity is helpful because ')]
            response_pro = random.choice(response_pro_list) + str(label2response[pred_label])
            dispatcher.utter_message(text=response_pro)

            if tracker.get_slot("pa_pro") is None:
                return [SlotSet("pa_pro", input_text)]
            else:
                pa_pro_raw = tracker.get_slot("pa_pro")
                pa_pro = pa_pro_raw + ' /sep ' + input_text
                return [SlotSet("pa_pro", pa_pro)]


class PA_ProReasonReflection(Action):

    def name(self) -> Text:
        return "action_pa_pro_reason_reflection"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        current_path = os.getcwd()
        save_model_path = current_path+'/ask_reason/model'

        pa_pro = tracker.get_slot("pa_pro").split(' /sep ')
        for i in pa_pro:
            pred_label = prediction.predict(save_model_path, i)
            response_pro_reflection = 'You said ' + '"' + str(i) + '", ' + 'which I think is related to ' + str(pred_label)
            dispatcher.utter_message(text=response_pro_reflection)

        return []


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


class AskRatingImportanceContinue(Action):

    def name(self) -> Text:
        return "action_rating_importance_judge"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        importance_rate = tracker.get_slot("importance_rate")
        print(f'importance_rate is: {importance_rate} - {type(importance_rate)}')

        if int(importance_rate) < 2:
            print(f"case 1")
            dispatcher.utter_message(text=f"It seems like you don't think physical activity is that important to you, becasue you rate the importance as {importance_rate}.")
            return [FollowupAction(name='rating_importance_low_importance_form')]
        
        elif int(importance_rate) > 8:
            print(f"case 2")
            dispatcher.utter_message(text=f"Wow, you rate the importance very high as {importance_rate}!")
            dispatcher.utter_message(text=f"Great! It seems like you already find that physical activity is extremely important to you.")
            return [FollowupAction(name='utter_rating_importance_ask_if_show_video')]
        
        else:
            print(f"case 3")
            return [FollowupAction(name='rating_importance_medium_importance_form')]


class AskRatingImportanceLessReason(Action):

    def name(self) -> Text:
        return "action_ask_rating_importance_less_reason"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        importance_rate = tracker.get_slot("importance_rate")
        importance_rate_less = int(importance_rate) - 2

        # dispatcher.utter_message(text=f"Ok, so you rate the importance as {importance_rate}.")
        dispatcher.utter_message(text=f"Why are you at a {importance_rate} but not a {importance_rate_less}?")
        dispatcher.utter_message(text=f"Tell me one reason why you think PA is important to you. Keep going!")

        return []


class AskRatingImportanceMoreReason(Action):

    def name(self) -> Text:
        return "action_ask_rating_importance_more_reason"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        rating_importance_less_reason = tracker.get_slot("rating_importance_less_reason")
        print(f'rating_importance_less_reason is: {rating_importance_less_reason}')

        importance_rate = tracker.get_slot("importance_rate")
        importance_rate_more = int(importance_rate) + 2
        
        # reflection1 = reflection(rating_importance_less_reason)
        # reflection1 = parap.reflection(rating_importance_less_reason)
        # response1 = f"That's something that could help you with challenge, since you said: {rating_importance_less_reason}."

        response2 = f"Why did you rate it a {importance_rate} and not a {importance_rate_more}?"
        response3 = f"What would it take to rate it a {importance_rate_more} and make PA  more important to you?"
        
        if rating_importance_less_reason != None:
            reflection = nlg_plugin(tracker, dispatcher) 
            dispatcher.utter_message(text=response2)
            dispatcher.utter_message(text=response3)
        else:
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

        importance_rate_less = int(importance_rate) - 2
        importance_rate_more = int(importance_rate) + 2

        # reflection = f"Ok, I can see that. Sounds reasonable indeed."
        reflection = nlg_plugin(tracker, dispatcher) 

        response2 = f"Now I understand what can make you rate more, because you said: {rating_importance_more_reason}."
        dispatcher.utter_message(text=response2)
        
        return []


##################################################################################################


# This custom action is for rating_confidence_form
class AskRatingConfidence(Action):

    def name(self) -> Text:
        return "action_ask_confidence_rate"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        step_count_goal = tracker.get_slot("step_count_goal")

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

        dispatcher.utter_message(text=f'On a scale from 0 to 10, where 0 is not at all confident and 10 is extremely confident, how confident are you that you will reach {step_count_goal} steps tomorrow?',
                                buttons=buttons)      
        return []


class AskRatingConfidenceContinue(Action):

    def name(self) -> Text:
        return "action_rating_confidence_judge"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence_rate = tracker.get_slot("confidence_rate")
        print(f'confidence_rate is: {confidence_rate} - {type(confidence_rate)}')

        if int(confidence_rate) < 2:
            print(f"case 1")
            dispatcher.utter_message(text=f"Seems like you're not very confident about your PA goal, becasue you rate the confidence as {confidence_rate}.")
            return [FollowupAction(name='rating_confidence_low_confidence_form')]
        
        elif int(confidence_rate) > 8:
            print(f"case 2")
            dispatcher.utter_message(text=f"Wow, you rate the confidence very high as {confidence_rate}!")
            dispatcher.utter_message(text=f"Seems like you're already very confident about your PA goal. Great, let's do that!")
            return [FollowupAction(name='action_connector_change_topic_model')]
        
        else:
            print(f"case 3")
            return [FollowupAction(name='rating_confidence_medium_confidence_form')]


class AskRatingConfidenceLessReason(Action):

    def name(self) -> Text:
        return "action_ask_rating_confidence_less_reason"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence_rate = tracker.get_slot("confidence_rate")
        confidence_rate_less = int(confidence_rate) - 2

        dispatcher.utter_message(text=f"Ok, so you rate the confidence as {confidence_rate}.")
        dispatcher.utter_message(text=f"Why are you at a {confidence_rate} but not a {confidence_rate_less}?")
        dispatcher.utter_message(text=f"Tell me one strength of you that makes you confident in achieving your daily step count goal.")

        return []


class AskRatingConfidenceMoreReason(Action):

    def name(self) -> Text:
        return "action_ask_rating_confidence_more_reason"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        rating_confidence_less_reason = tracker.get_slot("rating_confidence_less_reason")
        confidence_rate = tracker.get_slot("confidence_rate")
        confidence_rate_more = int(confidence_rate) + 2
        
        # reflection = parap.reflection(rating_importance_less_reason)
        # reflection = f"That's indeed something could help you with challenges. That is a good quality to have."

        response2 = f"Why did you rate it a {confidence_rate} and not a {confidence_rate_more}?"
        response3 = f"What would make you more confident in achieving your step count goal and make you rate your confidence a {confidence_rate_more}?"

        if rating_confidence_less_reason != "not_given":
            reflection = nlg_plugin(tracker, dispatcher) 
            dispatcher.utter_message(text=response2)
            dispatcher.utter_message(text=response3)
        else:
            dispatcher.utter_message(text=response2)
            dispatcher.utter_message(text=response3)

        return []


class RatingConfidenceStop(Action):

    def name(self) -> Text:
        return "action_rating_confidence_stop"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        rating_confidence_more_reason = tracker.get_slot("rating_confidence_more_reason")
        confidence_rate = tracker.get_slot("confidence_rate")
        # print(f'rating_confidence_more_reason is: {rating_confidence_more_reason}')

        confidence_rate_less = int(confidence_rate) - 2
        confidence_rate_more = int(confidence_rate) + 2

        response1 = f"Ok, I know what can make you rate more, because you said: {rating_confidence_more_reason}."
        dispatcher.utter_message(text=response1)

        # reflection = f"Sound reasonable indeed! Let's try and focus on that"
        reflection = nlg_plugin(tracker, dispatcher) 
        
        return []


##################################################################################################


# # This custom action is for self_efficacy_form
# class SelfEfficacyForm(FormAction):
#     """Example of a custom form action"""

#     def name(self):
#         """Unique identifier of the form"""
#         return "self_efficacy_form"

#     @staticmethod
#     def required_slots(tracker: Tracker) -> List[Text]:
#         """A list of required slots that the form has to fill"""
#         return ["self_efficacy_q1", "self_efficacy_q2"]

#     def submit(self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
#         ) -> List[Dict[Text, Any]]:
#         """Define what the form has to do after all required slots are filled"""
#         dispatcher.utter_message(text=f"Thanks for answering!")
#         return []


# # This custom action is for self_efficacy_form
class AskEfficacyRecall(Action):

    def name(self) -> Text:
        return "action_ask_efficacy_recall"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        response1 = f"Let's think of a moment in time where you ultimately succeeded in doing something very difficult."
        response2 = f"What was the task? Or what did you have to do?"
        
        dispatcher.utter_message(text=response1)
        dispatcher.utter_message(text=response2)
  
        return []


class AskReasonChallengeSuccess(Action):

    def name(self) -> Text:
        return "action_ask_efficacy_reason_for_challenge_success"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # reflection = f"That does sound challenging."
        reflection = nlg_plugin(tracker, dispatcher)

        response2 = f"How did you ultimately succeed?"
        dispatcher.utter_message(text=response2)
        
        return []


class AskFeelChallenge(Action):

    def name(self) -> Text:
        return "action_ask_efficacy_feel_of_challenge"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # reflection = f"That's indeed one thing can help you go through challenging times."
        # response_affirmation = f"You seem very resilient."
        reflection = nlg_plugin(tracker, dispatcher)

        response2 = f"Now let's revisit that and talk more about how you felt. How did the challenge make you feel? "
        dispatcher.utter_message(text=response2)

        return []


class AskRemediateChallenge(Action):

    def name(self) -> Text:
        return "action_ask_efficacy_remediate_challenge"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        # reflection1 = f"You had hard times, I can imagine. But you managed to succeed in the end!"
        reflection = nlg_plugin(tracker, dispatcher)
        
        response2 = f"I wonder, is there something you can think of that remediated these feelings?"
        dispatcher.utter_message(text=response2)

        return []


class AskFurtherEfficacyTalk(Action):

    def name(self) -> Text:
        return "action_efficacy_further_talk"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        
        # reflection = f"Sounds like a good way for you to remediated these feelings!"
        reflection = nlg_plugin(tracker, dispatcher)
        
        response2 = f"Thanks for your sharing so far!"
        dispatcher.utter_message(text=response2)

        buttons = [
            {"payload": '/trigger_self_efficacy_further', "title": "Sure!"},
            {"payload": '/deny', "title": "No, I don't want to."}]
        
        dispatcher.utter_message(text=f'Shall we go further and talk about that?', buttons=buttons)      
        
        return []


class AskLearntFromChallenge(Action):

    def name(self) -> Text:
        return "action_ask_efficacy_learnt_from_challenge"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        response1 = f"I'm curious how you make sense of this experience, what did you learn?"
        
        dispatcher.utter_message(text=response1)
        
        return []


class AskEfficacyCompleteReflection(Action):

    def name(self) -> Text:
        return "action_efficacy_complete_reflection"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
       
        # response1 = f"Great to hear that you learned something from the challenge! And I agree that it is the thing you can learn from the challenge."
        # response2 = f"I appreciate you telling me about your previous challenges and your feelings regarding these challenges."
        # response3 = f"I do think you can learn a lot after you've experienced a challenge."

        reflection = nlg_plugin(tracker, dispatcher)
        
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

        # reflection = f'It sounds like you are pretty happy with your current level of physical activity!'
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

        # reflection = f'I see how that can cause some negative feelings.'
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


class RetrievalProfile(Action):

    def name(self) -> Text:
        return "action_retrieval_db_profile"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
       
        pass

        return []


##################################################################################################


class RetrievalGoal(Action):

    def name(self) -> Text:
        return "action_retrieval_db_goal"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
       
        pass

        return []


##################################################################################################


class RetrievalDailyStepCount(Action):

    def name(self) -> Text:
        return "action_retrieval_daily_step_count"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
       
        daily_step = tracker.get_slot("daily_step")
        step_count_goal = tracker.get_slot("step_count_goal")
        update_goal_value = tracker.get_slot("update_goal_value")

        if update_goal_value == None:
            if int(daily_step) > int(step_count_goal):
                response = f'Congratulions! Today you already did more steps than your goal. Your goal was {step_count_goal} and the amount of steps you did was {daily_step}.' 
            else:
                response = f'Your step count goal was {step_count_goal} and you already did {daily_step} steps. That is already a good start.'
            dispatcher.utter_message(text=response)
        else:
            if int(daily_step) > int(update_goal_value):
                response = f'Congratulions! Today you already did more steps than your goal. Your goal was {update_goal_value} and the amount of steps you did was {daily_step}.' 
            else:
                response = f'Your step count goal was {update_goal_value} and you already did {daily_step} steps. That is already a good start.'
            dispatcher.utter_message(text=response)

        return []



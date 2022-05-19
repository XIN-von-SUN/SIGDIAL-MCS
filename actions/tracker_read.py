import json
import os

# This function read all tracker information: tracker.events_after_latest_restart() and transfor all dialogue record into the pure dialogue history txt format
def tracker_read(tracker_file, pure_dialog_records):
    df = []
    # tracker = open(tracker_file)
    with open(tracker_file,'r') as tracker_file:
        tracker = json.load(tracker_file)
        print(tracker)
        # print(type(tracker))
        for event in tracker:
            if event['event'] == "user":
                if event['parse_data']['entities'] == []:
                    df.append({'user input':event['text'], 'intent':event['parse_data']['intent']['name'], 'sentiment':"neu"})
                else:
                    df.append({'user input':event['text'], 'intent':event['parse_data']['intent']['name'], 'sentiment':event['parse_data']['entities'][0]['value']})

            elif event['event'] == "bot":
                df.append({'bot reply':event['text']})

        tracker_file.close()
    
    with open(pure_dialog_records,'w+') as pure_dialog_records:
        for record in df:
            pure_dialog_records.write(str(record) + '\n')
        pure_dialog_records.close()

if __name__ == '__main__': 
    current_path = os.getcwd()
    previous_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    tracker_file = previous_path + '/data/dialogue_history/events_after_latest_restart.json'
    pure_dialog_records = previous_path + '/data/dialogue_history/pure_dialog_records.txt'
    
    tracker_read(tracker_file, pure_dialog_records)
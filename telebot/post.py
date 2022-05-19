import requests

url_post = "http://1779-145-38-196-223.ngrok.io/webhooks/rest/webhook"

# myobj = {"sender": "t11", "message": "hi"}


def post_user(sender_id, url_post, user_utter, data_json):

    response = requests.post(url_post, json=data_json)

    return response.json() #.text


def get_response_tracker(url_response):

    tracker = requests.get(url_response)

    return tracker.json()


def run(sender_id, url_post):
    user_utter = input("user saying: ")
    while user_utter != "stop":
        data_json = {"sender": sender_id, "message": user_utter}

        url_response = (
            f"http://1779-145-38-196-223.ngrok.io/conversations/{sender_id}/tracker"
        )
        tracker = get_response_tracker(url_response)
        print(f"url_response: {url_response}\n")
        print(f"tracker: {tracker}\n")

        response = post_user(sender_id, url_post, user_utter, data_json)
        print(f"bot: {response}\n")
        user_utter = input()


if __name__ == "__main__":
    run("t001", url_post)

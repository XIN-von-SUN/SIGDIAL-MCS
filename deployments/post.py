import requests

url = 'http://fea3-145-38-196-223.ngrok.io/webhooks/rest/webhook'
myobj = {"sender": "t11", "message": "hi"}

x = requests.post(url, json=myobj)

print(x.text)

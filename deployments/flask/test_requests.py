import requests

user_info = {'name': 'letian', 'age': 12, 'gender':'M'}
r = requests.post("http://127.0.0.1:5000/test", data=user_info)

print(f'response is:{r.text}')


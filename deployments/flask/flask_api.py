from flask import Flask, redirect, url_for, request
import json

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'


# @app.route('/success/<name>')
# def success(name):
#     return 'welcome %s' % name

# @app.route('/login',methods = ['POST', 'GET'])
# def login():
#     if request.method == 'POST':
#         user = request.form['name']
#         return redirect(url_for('success',name = user))
#     else:
#         user = request.args.get('name')
#         return redirect(url_for('success',name = user))


# 从直接传入的user input中request获取数据
@app.route('/test1', methods = ['POST', 'GET'])
def test1(): 
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    return 'hello name=%s, age=%s, gender=%s' % (name, age, gender)
    # return json.dumps({'id':'1', 'meta':'a', 'text':'hey'})


# 从web输入的user input中request获取数据
@app.route('/test2',methods = ['POST', 'GET'])
def test2():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    return 'Hey! My info: name=%s, age=%s, gender=%s' % (name, age, gender)


if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)
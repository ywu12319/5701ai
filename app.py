from flask import Flask
import os
from flask import request

app = Flask(__name__)

global count

@app.route("/")
def hello_world():
    return "<p>Hello, World! good </p>"



@app.route("/submit_img",methods=["post"])
def submit_img():
    path = os.path.join(os.path.expanduser("~"),"Desktop","5701AI","image")
    print("path is" + path)
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    request.files["image"].save(os.path.join(path,"process.jpg"))
    return "sucess"



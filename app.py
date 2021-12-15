from flask import Flask
import os
from flask import request,render_template
from pytorch import Net
from pytorch import classify, imgRead
from torchvision import transforms
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

classification_net = Net()

transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.7369, 0.6360, 0.5318),
                                                           (0.3281, 0.3417, 0.3704))
                                      ])
classification_net.load_state_dict(torch.load("./fruit.pt",map_location=torch.device('cpu')))

model = tf.keras.models.load_model('./model')





@app.route("/",methods=['POST','GET'])
def hello_world():
    if request.method == "POST":
        image = request.files.get('img', '')
        image = request.form["img"]
        print("img ",image)
        
    else:
        return render_template('index.html') 



@app.route("/submit_img",methods=["post"])
def submit_img():
    path = os.path.join(os.path.expanduser("~"),"Desktop","5701AI","image")
    if not os.path.exists(path):
        os.makedirs(path)
    request.files["img"].save(os.path.join(path,"process.png"))
    print("type is")
    print(request.form['type'])
    if(request.form['type'] == 'fruit'):
        res = classify(imgRead('./image/process.png',transformations), classification_net)
        response = res.tolist()
        fresh_prob = response[0][0]
        rotten_prob = response[0][1]
        if(fresh_prob > rotten_prob):
            return {"res":"fresh"}
        else:
            return {"res":"rotten"} 
    else:
        image=load_img("./image/process.png",target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        prediction_image=np.array(image)
        prediction_image= np.expand_dims(image, axis=0)
        prediction=model.predict(prediction_image)
        value=np.argmax(prediction)
        print(value)
        if(value == 0):
            return {"res":"fresh"}
        else:
            return {"res":"rotten"} 
        


if __name__ == '__main__':
    app.run(debug=False)
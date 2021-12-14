from flask import Flask
import os
from flask import request
from pytorch import Net
from pytorch import classify, imgRead
from torchvision import transforms
import torch

app = Flask(__name__)

classification_net = Net()

transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.7369, 0.6360, 0.5318),
                                                           (0.3281, 0.3417, 0.3704))
                                      ])
classification_net.load_state_dict(torch.load("./fruit.pt",map_location=torch.device('cpu')))

@app.route("/")
def hello_world():
    return "<p>Hello, World! good </p>"



@app.route("/submit_img",methods=["post"])
def submit_img():
    path = os.path.join(os.path.expanduser("~"),"Desktop","5701AI","image")
    if not os.path.exists(path):
        os.makedirs(path)
    request.files["image"].save(os.path.join(path,"process.png"))

    res = classify(imgRead('./image/process.png',transformations), classification_net)
    response = res.tolist()
    fresh_prob = response[0][0]
    rotten_prob = response[0][1]
    if(fresh_prob > rotten_prob):
        return {"res":0}
    else:
        return {"res":-1}


if __name__ == '__main__':
    app.run(debug=False)
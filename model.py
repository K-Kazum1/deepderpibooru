import sys
import json
import torch
from PIL import Image
import pickle
import os
import numpy as np
import requests

from CLIP import clip
sys.modules['clip'] = clip #evil hack

device = "cuda" if torch.cuda.is_available() else "cpu"

print('loading tags')
taglist = open("./tags",'r').read().split('\n')[:-1]

def load_new():
    print('loading model')
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.to(torch.float32)

    linear = torch.nn.Linear(model.visual.output_dim,len(taglist)).to('cuda')
    linear.train(True)

    model = torch.nn.Sequential(model.visual,linear)

    return model

model, preprocess = clip.load("ViT-L/14@336px", device=device) #need preprocess
model = pickle.load(open('./model.torch','rb'))

def predict(image):
    if os.path.exists(image):
        image = Image.open(image)
    else:
        image = Image.open(requests.get(image, stream=True).raw)
    o = model(torch.unsqueeze(preprocess(image),0).to(device))[0]
    
    return ', '.join([taglist[i] for i in sorted(range(len(taglist)),key = lambda x:o[x],reverse=True)[:20]])


while True:
    print(predict(input()))



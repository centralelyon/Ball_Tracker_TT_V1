
from xml.sax.handler import DTDHandler
import csv
import glob 
import json
import os
import random
data_json = open('ball_marker.json','r')
data_ball = json.load(data_json)
train_set = open('train_set.csv','w')
test_set = open('test_set.csv','w')
writer = csv.writer(train_set)
writer2 = csv.writer(test_set)
for i in glob.glob('image/*.jpg'):
    char ='\image/.jpg'
    str=os.path.basename(i)
    str = ''.join(x for x in str if x not in char)
    if str in data_ball.keys():
        if random.random()>0.7:
            writer2.writerow([os.path.basename(i),1])
        else:
            writer.writerow([os.path.basename(i),1])
    else :
        if random.random()>0.7:
            writer2.writerow([os.path.basename(i),0])
        else:
             writer.writerow([os.path.basename(i),0])
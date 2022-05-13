from this import d
from xml.sax.handler import DTDHandler
import cv2
import glob 
import json
import os
data_json = open('ball_marker.json','r')
data_ball = json.load(data_json)
for i in glob.glob('image/*.jpg'):
    char ='\image/.jpg'
    str=os.path.basename(i)
    str = ''.join(x for x in str if x not in char)
    if str in data_ball.keys():
        cv2.imwrite(f'avec_balle/{int(str)}.jpg', cv2.imread(f'image/{int(str)}.jpg'))
    else :
        cv2.imwrite(f'sans_balle/{int(str)}.jpg', cv2.imread(f'image/{int(str)}.jpg'))        
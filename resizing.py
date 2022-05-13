import cv2
import glob
import os

for i in glob.glob('image/*.jpg'):
    chr = 'image\.jpg'
    str = ''.join(k for k in i if k not in chr) 
    if os.path.isfile(f'imager/{str}.jpg')==False:
    	img = cv2.imread(i)
    	resized_img = cv2.resize(img, (256,144))
    	cv2.imwrite(f'imager/{str}.jpg', resized_img)


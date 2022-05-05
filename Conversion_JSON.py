import csv
import json

data_csv = open ('resultat_tracker.csv','r')
data_json = open('ball_marker.json','w')

reader = csv.reader(data_csv,delimiter=';')
next(reader)

dict_json={}
for i in reader:
    if i[1]!='':
        dict_json[i[3]]={"x":float(i[1].replace(',','.')),"y":float(i[2].replace(',','.'))}
json.dump(dict_json,data_json)

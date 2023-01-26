import torch, random, os, json
from collections import OrderedDict, Counter



path = "/Users/bogdankhamelyuk/Documents/FLIR_ADAS_v2/video_thermal_test/index.json"


total_person_number = 0
annotation_list = []

with open(file=path,mode="r") as file:
    json_file = json.load(file)
    
    for annotations in json_file["frames"]:

        #list = OrderedDict()
        frameID = annotations["datasetFrameId"]
        list = {}
        list["id"] = frameID
        list["coordinates"] = []
        for elements in annotations["annotations"]:
            
            if elements["labels"][0] == "person":
                
                print(elements["boundingBox"])
                x_ctr = elements["boundingBox"]["x"] + elements["boundingBox"]["w"]/2
                y_ctr = elements["boundingBox"]["y"] + elements["boundingBox"]["h"]/2
                #list[frameID].append({x_ctr,y_ctr})
                #temp1 = Counter({"x":x_ctr,"y":y_ctr})
                coordinates = [x_ctr,y_ctr]
                list["coordinates"].append(coordinates)
        if len(list["coordinates"])>0:
            annotation_list.append(list)
    
    print(total_person_number)
file_name = "FLIR_video_labels.json"
path = "/Users/bogdankhamelyuk/Documents/JKU_MS_AI/computer_vision/"  + file_name
with open(file=path,mode="w+") as new_json_file:
    json.dump(annotation_list,new_json_file,indent=4)



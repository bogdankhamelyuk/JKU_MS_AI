import json
import torch
import time
import random 
import os

class Dataset:
    hello = ""
    def __init__(self):
        Dataset.print_hello(Dataset.hello)
    
    @classmethod
    def print_hello(cls,abc):
        abc="hello"
        print(abc)

a = Dataset()


training_file_names, test_file_names = [], []
start = time.time()
random.seed(0xDEADBEEF)
i = 0
while i < 20:
    r = random.randint(0,27)
    if not r in training_file_names and not r in test_file_names:
        training_file_names.append(r)
        i+=1
i = 0
while i < 7:
    r = random.randint(0,27)
    if not r in training_file_names and not r in test_file_names:
        test_file_names.append(r)
        i+=1  
    

end = time.time()
diff = end - start
for i in range(7):
    r = random.randint(0,27)
    if r not in (training_file_names and test_file_names):
        test_file_names.append(r)
    else:
        i-=1
path = os.getcwd()

video_name = "video_" + str(r) + ".json"
video_labels = {"video_number":video_name, "labels":[]}
vid_cont = open(file=path+"/computer_vision/labels/"+video_name, mode="r")
vid_cont_json = json.load(vid_cont)[str(r)]

for frame in vid_cont_json:
    grid = torch.zeros(16,16,3)
    for recognized_person in frame:
        row, column, x, y = frame[recognized_person] 
        grid[int(row)][int(column)][0]=1
        grid[int(row)][int(column)][1]=x
        grid[int(row)][int(column)][2]=y
    video_labels["labels"].append(grid)    

video_labels["labels"] = torch.stack(video_labels["labels"])
for frame in video_labels["labels"]:
    pass
pass    
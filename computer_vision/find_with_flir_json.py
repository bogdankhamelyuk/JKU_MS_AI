import torchvision, os,json, torch
path = "/Users/bogdankhamelyuk/Documents/FLIR_ADAS_v2/images_thermal_train/data/"
labels = json.load(open("/Users/bogdankhamelyuk/Documents/JKU_MS_AI/computer_vision/FLIR_train_labels.json"))
l = len(labels)
label = labels[0]
max_coordinates= 0

for label in labels:
    img_id = label["id"]
    for im in os.listdir(path):
        if im.endswith(img_id+".jpg"):
            image = torchvision.io.read_image(path+im)
            #print(image.shape)
    
    labels = []  

    # create tensor for the x,y-coordinates of the label
    i = 0
    for labeled_person in label["coordinates"]:
        
        x_abs = labeled_person[0]
        y_abs = labeled_person[1]
        
        # print([x_abs,y_abs])
        labels.append(torch.tensor([x_abs,y_abs])) # append that 16x16x3 gridded frame to the labels
        i+=1
    labels = torch.stack(labels)
    if i>max_coordinates:
        max_coordinates=i
        great_labels = labels
        print("LAST MAX COORD: ",max_coordinates)
        if i>59:
            print(labels)
    
    
pass
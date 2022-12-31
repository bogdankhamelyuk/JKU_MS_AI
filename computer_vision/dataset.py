import torch, torchvision, os, random, json
from torch.utils.data import Dataset, DataLoader

path = os.getcwd() + "/computer_vision/labels/"

def count_all_jsons(): 
    total_json_number = 0
    for file in os.listdir(path):
        if file.endswith('.json'):
            total_json_number+=1
    return total_json_number

class DroneThrmImg_Dataset(Dataset):
    # create arrays, where random and non-repeating numbers of videos in the working directory will be stored 
    train_file_names, test_file_names = [], [] 
    random.seed(10)

    def __init__(self,req_samples_number, dataset_type):
        self.dataset_type = dataset_type
        self.total_json_number = count_all_jsons()
        
        if req_samples_number < self.total_json_number:
            self.req_samples_number = req_samples_number # number of samples required for training/test dataset and is less than total number of samples
        else:
            raise IndexError(f"required samples number ({req_samples_number}) is greater than\
 total number of available samples({self.total_json_number})")
        
        if self.dataset_type == 'TEST':
            self.random_videos = self.append_random_number(DroneThrmImg_Dataset.test_file_names,self.req_samples_number)

        elif self.dataset_type == 'TRAIN':
            self.random_videos = self.append_random_number(DroneThrmImg_Dataset.train_file_names,self.req_samples_number)

        else:
            raise NameError(f"'{dataset_type}' is not in the list of available dataset types. \nAvailable types are: 'TEST' and 'TRAIN.")
        

    def __len__(self):
        return self.req_samples_number # return required number of samples, since that's exactly length of the each dataset

    def __getitem__(self, idx):
        video_number = self.random_videos[idx] # take out random generated number of the video, according to its index
        label_name = "video_" + str(video_number) + ".json" # append this number to "video_" and ".json", like all other videos are named
        video_name = "video_" + str(video_number) + ".mp4"
        label_path = path+label_name # create complete path to the indexed video, i.e. "labels" of that video
        labels = open(file=label_path, mode="r")
        labels_json = json.load(labels)[str(video_number)] # adjust output of the json using [str(video_number)], to access labels
        video_tensor = torchvision.io.read_video(path+video_name)[0] # create video tensor for the output
        #print(video_tensor.shape)
        output = {"video":video_tensor, "labels":[]} # create output return dictionary containing video name and labels for each frame

        for labeled_frame in labels_json: # each labeled frame in json has number of recogn. people and their coordinates, i.e. coordinates of label
            # create 16x16x3 grid, in each position of which is
            # the prob-ty of recognition and x,y-coordinates of the label to be stored
            grid = torch.zeros(16,16,3)             
            # since the number of people can vary 
            # make it sure that all labels in the grid are stored                 
            for recognized_person in labeled_frame: 
                row, column, x, y = labeled_frame[recognized_person]
                grid[int(row)][int(column)][0]=1 # probability that in this grid is person, make it to 100%
                grid[int(row)][int(column)][1]=x # x-coordinate of the label
                grid[int(row)][int(column)][2]=y # y-coordinate of the label
            output["labels"].append(grid) # append that 16x16x3 gridded frame to the labels
        output["labels"] = torch.stack(output["labels"]) # convert labels to the torch-type tensor
        #print(output["labels"].shape)
        return output 

            
    def append_random_number(self, optional_list, samples_number):
        i = 0
        while i < samples_number:
            # substract 1 from total_json_number, because the first video is named using 0, not 1, 
            # so the number of the last video is 1 less then total quantity
            r = random.randint(0,self.total_json_number-1) 
            if not r in DroneThrmImg_Dataset.train_file_names \
                        and not r in DroneThrmImg_Dataset.test_file_names:
                optional_list.append(r)
                i+=1
        return optional_list


all_files = count_all_jsons()
train_files = int(0.8*all_files)
test_files = int(0.2*all_files)

test_dataset = DroneThrmImg_Dataset(req_samples_number=test_files,  dataset_type='TEST')
train_dataset = DroneThrmImg_Dataset(req_samples_number=train_files, dataset_type='TRAIN')


train_dataloader = DataLoader(train_dataset,shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True)

output = next(iter(train_dataloader))
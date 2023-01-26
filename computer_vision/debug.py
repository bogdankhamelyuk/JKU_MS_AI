import json
import torch, torchvision, os, random
from torch.utils.data import Dataset, DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")


label_path = "/Users/bogdankhamelyuk/Documents/JKU_MS_AI/computer_vision/labels/"
video_path = "/Users/bogdankhamelyuk/Documents/JKU_MS_AI/computer_vision/videos/"
def count_all_jsons():
    """
    Calculate number of available json-files in the dataset.

    Returns
    ---
    total_json_number: int 
        Total number of json files inside of label path 
    """
    total_json_number = 0
    for file in os.listdir(label_path):
        if file.endswith('.json'):
            total_json_number+=1
    return total_json_number

class DroneThrmImg_Dataset(Dataset):
    """
    Custom dataset class, which inherits from PyTorch's `Dataset`. 

    Attributes
    ---
    train_file_names : list
        List in which random number of train videos will be stored.
    test_file_names: list
        List in which random number of test videos will be stored.

    Methods
    ---
    1) `__init__` - used for initialization of the dataset class;
    2) `__len__` - returns length of the dataset;
    3) `__getitem__`- returns tensor of video and corresponding label according to the index in the list;
    4) `append_random_number` - appends random number in the range of 0 and total amount of json files. 
    """
    train_file_names, test_file_names = [], [] 
    random.seed(10)

    def __init__(self,req_samples_number: int, dataset_type: str):
        """
        Initializes dataset. 

        Parameters
        ---
        dataset_type : str
            Type of dataset, which must be either 'TEST' or 'TRAIN'.

        req_samples_number : int
            Number of samples required for that dataset.
        """
        self.dataset_type = dataset_type
        self.total_json_number = count_all_jsons()

        if req_samples_number < self.total_json_number:
            self.req_samples_number = req_samples_number 
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
        """
        Returns length of dataset, which is specified by user through `req_samples_number` parameter
        """
        return self.req_samples_number 

    def __getitem__(self, idx: int):
        """
        Gets index of the video and prepares video with its label, by looking into provided paths. 
        Returns tensor of the video and corresponding label. 

        Parameters
        ---
        idx : int

        Returns
        ---
        gray_video_tensor : torch.float32
            Random video that matches with its index in the list and that has following dimensions: T,C,W,H.
                T - Number of frames,
                C - Number of channels (always 1, because it's grayscaled),
                W - Width of the frame, which is 256 in the provided dataset,
                H - Height of the frame, which is 320 in the provided dataset.

        labels : torch.float32
            Stacked labels of the video, with the dimensions: T,1,2.
    
        """
        # prepare videos, which are available in the local directory
        video_number = self.random_videos[idx] # take out random generated number of the video, according to its index
        self.video_name = "video_" + str(video_number) 
        video_tensor = torchvision.io.read_video(video_path+self.video_name+".mp4")[0] # create video tensor for the output, which shape 
       
        # T,H,W,C = video_tensor.size()
        # 0,1,2,3
        # 2415, 256, 320, 3 (for example)
        gray_video_tensor = video_tensor.narrow(-1,0,1) # narrow last dimension(=channels) to 1
        # print(gray_video_tensor.size())
        gray_video_tensor = torch.permute(gray_video_tensor,(0,3,1,2))
        # print(gray_video_tensor.size())

        # open corresponding label of the video
        labeled_video = json.load(open(label_path+self.video_name+".json"))

        labels = []
        for labeled_frame in labeled_video:
            grid = torch.zeros(2)             
            for objects in labeled_frame:
                x_abs = objects['x']
                y_abs = objects['y']
                grid[0]=x_abs 
                grid[1]=y_abs 
                labels.append(grid)
        labels = torch.stack(labels) # convert labels to the torch-type tensor
        
        # convert tensors to torch float 32 and locate them on device
        labels = labels.to(device).to(torch.float32)
        gray_video_tensor = gray_video_tensor.to(device).to(torch.float32)

        return gray_video_tensor, labels 
  
    def append_random_number(self, optional_list: list, samples_number: int):
        """
        Appends random number to the provided list in the parameters.

        Parameters
        ---
        optional_list : list 
            List in which random number will be stored

        Returns
        ---
        optional_list : list 
            Same provided list in the parameters, with the random video numbers inside of it.
        """
        i = 0
        while i < samples_number:
        # repeat till all number will be really random and satisfy required amount
            # substract 1 from total_json_number, because the first video starts from 0, not 1, 
            # so the number of the last video is 1 less then total quantity
            r = random.randint(0,self.total_json_number-1) 
            # check whether this number of the video is already used by another list
            if not r in DroneThrmImg_Dataset.train_file_names \
                        and not r in DroneThrmImg_Dataset.test_file_names:
                optional_list.append(r)
                # go step further only if random video isn't another list
                i+=1

        return optional_list
all_files = count_all_jsons()
train_files = int(0.9*all_files)
test_files = all_files - train_files

test_dataset = DroneThrmImg_Dataset(req_samples_number=test_files,  dataset_type='TEST')
train_dataset = DroneThrmImg_Dataset(req_samples_number=train_files, dataset_type='TRAIN')


class simpleModel(torch.nn.Module):
    def __init__(self):
        self.idx = 0
            
        super(simpleModel,self).__init__()
        self.input_channels = 1
        
        self.hidden_size = 10

        self.conv1 = torch.nn.Conv2d(1,40, kernel_size=(6,5),stride=(5),padding=(0))    # --> 51,64
        self.conv2 = torch.nn.Conv2d(40,40, kernel_size=(5,6),stride=(2),padding=(1))  # --> 25,31
        self.conv3 = torch.nn.Conv2d(40,120, kernel_size=(4),stride=(3),padding=(0)) # 8,10
        #self.bn = torch.nn.BatchNorm1d(8)
        self.relu0 = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2))# -->4,5
        
        self.flatten1 = torch.nn.Flatten()

        input_size = 20
        hidden_size = 175
        seq_length = 120 

        self.lstm = torch.nn.LSTM(input_size,hidden_size,num_layers=2)
        
        self.flatten2 = torch.nn.Flatten(start_dim=0)
        self.linear3 = torch.nn.Linear(seq_length*hidden_size,2)
       
  
    def forward(self,frame):
        frame = self.conv1(frame)
        frame = self.conv2(frame)
        frame = self.conv3(frame)
    
        frame = self.relu0(frame)
        

        frame = self.maxpool(frame)
        frame = self.flatten1(frame)
        if self.idx == 0:
            output, (h_n, c_n) = self.lstm(frame)
            self.hidden = h_n.detach()
            self.cell = c_n.detach()
        else:
            output, (self.hidden,self.cell) = self.lstm(frame,(self.hidden,self.cell))
            self.hidden = self.hidden.detach()
            self.cell = self.cell.detach()

        frame = self.flatten2(output)
        coordinates = self.linear3(frame)
        self.idx+=1
        return coordinates
model = simpleModel().to(device)
loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


PATH = "/Users/bogdankhamelyuk/Documents/JKU_MS_AI/computer_vision/my_model.pth"

train_dataloader = DataLoader(train_dataset,shuffle=True)

model = torch.load(PATH)

diff_dict = {"<1px": 0,"1-2px": 0, "2-4px":0, "4-8px": 0, ">8px":0}

with torch.no_grad():
    for i, batch in enumerate(iter(train_dataloader)):
        print("iteration: ", i)
        video, label = batch
        video = video[0]
        label = label[0]
        model.idx = 0

        for frame, labeled_frame in zip(video,label):
            model.eval()
            output = model(frame)
            print(frame)
            
            print("predict: ",output)   
            print("target: ", labeled_frame)
            diff = torch.sum((abs(output-labeled_frame)))
            print(diff)
            

from django.db import models
import torch
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print("Device name: ", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("There are no GPU(s) available, using the CPU instead.")

# Create your models here.

class QADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # print(self.labels['input_ids'][idx])
        # print(torch.tensor(self.labels['input_ids'][idx]))
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    
    def __len__(self):
        return len(self.labels['input_ids'])
    
class ChatDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs.tolist()
        self.targets = targets.tolist()

        for idx, i in enumerate(self.inputs):
            try:
                self.inputs[idx] = "<SOS> " + self.inputs[idx] + " <BOT>: " + self.targets[idx+1] + " <EOS>"
            except:
                break
        
        # self.inputs = self.inputs[:-1]

        print(self.inputs[0])
        # print(len(self.inputs))

        self.inputs_encoded = tokenizer(self.inputs, truncation=True, padding=True, max_length=200, return_tensors='pt')
        self.labels_encoded = tokenizer(self.targets, truncation=True, padding=True, max_length=200, return_tensors='pt')
        self.input_ids = self.inputs_encoded['input_ids']
        self.attention_mask = self.inputs_encoded['attention_mask']
        self.label_input_ids = self.labels_encoded['input_ids']
        

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from helper import mytokenizer
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
class COCODataset(data.Dataset):
    def __init__(self,path, datadict, transforms=None):
        self.datadict = datadict
        self.transforms = transforms
        self.path = path
        self.vocab = Vocabulary(vocab_threshold = 1, vocab_from_file= True)
        self.img_list = list(self.datadict.items())
        captions_list = list(datadict.values())
        self.caption_lengths = []
        self.tokens_list = []
        for cap in tqdm(captions_list):
            tokens = mytokenizer(cap["captions"][4])
            self.caption_lengths.append(len(tokens))
            self.tokens_list.append(tokens)
            
    def __getitem__(self, idx):
        img_id, img_anno = self.img_list[idx]
        img_name = img_anno["file_name"]
        img = Image.open(os.path.join(self.path,img_name)).convert('RGB')
        
        captions = self.tokens_list[idx]
        
        captions_idx = []
        captions_idx.append(self.vocab(self.vocab.start_word))
        captions_idx.extend([self.vocab(i) for i in captions])
        captions_idx.append(self.vocab(self.vocab.end_word))
        captions_length = torch.Tensor([len(captions_idx)])
   
        captions_idx = torch.LongTensor(captions_idx)
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        img_item = {}
        img_item["img"] = img
        img_item["captions"] = " ".join(captions)
        img_item["captions_idx"] = captions_idx
        img_item["caption_lengths"] = self.caption_lengths[idx]
        return img_item
    
    def clean_sentence(self, output):
        sentence = []
        for i in output:
            sentence.append(self.vocab.itos(i))
            if i == self.vocab(self.vocab.end_word):
                break
        sentence = " ".join(sentence[1:-1])
        sentence = sentence.capitalize()
        return sentence
    
    
    def imshow(self,item,train=True):
        img = item["img"]
        caps = item["captions"]
        idx = item["captions_idx"]
        length = item["caption_lengths"]
        if train == False:
            img = img.squeeze()
        np_img = np.array(img).transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = np_img*std + mean
        img_show = np.clip(np_img,0,1)
        return img_show, caps
 
    
    def get_train_indices(self, BATCH_SIZE):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=BATCH_SIZE))
        return indices
    
    def __len__(self):
        return len(self.datadict)
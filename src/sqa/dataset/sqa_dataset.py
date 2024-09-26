import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import pandas as pd
import yaml
import lmdb
import io
import time
from vidaug import augmentors as va

import copy
from sqa.llama import Tokenizer, format_prompt

import pytorch_lightning as pl

# global definition
from sqa.dataset import utils
from sqa.dataset.augmentation import Brightness, Color
from sqa.dataset.definition import *





class SQA_Dataset(Dataset):

    def __init__(self, path, tokenizer_path, config, phase, max_words=85, training_refurbish=False, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words # max number of text words

        self.training_refurbish = training_refurbish

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = pd.read_csv(path, delimiter='|')
        # print(self.raw_data['train/11August_2010_Wednesday_tagesschau-1'])
        # exit()
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']    # max number of frames
        
        # self.list = [key for key,value in self.raw_data.items()]   

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10)),

        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
        ])
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        sample = self.raw_data.iloc[index]
        # print(sample['name'])
        file_name = sample['name']
        question = sample['question']
        answer = sample['answer']
        # key = self.list[index]
        # file_name = sample['imgs_path']
        input_1 = format_prompt(question)
        input_2 = input_1 + answer

        input_1 = torch.tensor(self.tokenizer.encode(input_1, bos=True, eos=False), dtype=torch.int64)
        input_2 = torch.tensor(self.tokenizer.encode(input_2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input_2.shape[0]
        if padding > 0:
            input_2 = torch.cat((input_2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input_2 = input_2[:self.max_words]
        
        labels = copy.deepcopy(input_2)
        labels[:len(input_1)] = -1
        input_2_mask = input_2.ge(0)
        label_mask = labels.ge(0)
        input_2[~input_2_mask] = -1
        labels[~label_mask] = -1
        input_2_mask = input_2_mask.float()
        label_mask = label_mask.float()

        img_sample = self.load_imgs(file_name)
        
        return file_name, img_sample, input_2, labels, input_2_mask
    
    def load_imgs(self, file_name):
        folder = os.path.join(self.lmdb_path, self.phase)
        images = utils.read_lmdb_folder(folder, file_name)
        len_imgs = len(images)
        
        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        
        if len_imgs > self.max_length:
            tmp = sorted(random.sample(range(len_imgs), k=self.max_length))
            new_images = []
            for i in tmp:
                new_images.append(images[i])
            images = new_images
    
        imgs = torch.zeros(len_imgs,3, self.input_size,self.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.resize, self.resize), crop_size=self.input_size, is_train=(self.phase=='train'))

        batch_image = []
        for i,img in enumerate(images):
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
        
        return imgs

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
    

if __name__ == "__main__":
    import yaml
    config = {
        'data': {
            'lmdb_path': 'src/sqa/data/lmdb',
            'max_length': 300,
        }
    }
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    tokenizer_path = 'src/sqa/llama/llama_dir/tokenizer.model'
    root_text_path = 'src/sqa/data/labels'
    phase = 'train'
    path = 'src/sqa/data/clean-qa.csv'
    dataset = SQA_Dataset(path=path, tokenizer_path=tokenizer_path, config=config, phase=phase)
    print(len(dataset))



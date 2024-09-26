from ctypes import util
from cv2 import IMREAD_GRAYSCALE
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
from sqa.dataset.sqa_dataset import SQA_Dataset

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset):

    def __init__(self, path, tokenizer_path, config, phase, max_words=85, training_refurbish=False, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words
        self.training_refurbish = training_refurbish

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = utils.load_dataset_file(path)
        # print(self.raw_data['train/11August_2010_Wednesday_tagesschau-1'])
        # exit()
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        
        self.list = [key for key,value in self.raw_data.items()]   

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
        key = self.list[index]
        sample = self.raw_data[key]
        file_name = sample['name']
        # file_name = sample['imgs_path']
        answer = sample['text']
        length = sample['length']
        input_1 = format_prompt()
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
        # print(img_sample.shape)
        
        return file_name, img_sample, input_2, labels, input_2_mask
    
    def load_imgs(self, file_name):
        phase, file_name = file_name.split('/')
        folder = os.path.join(self.lmdb_path, phase)
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


def collate_fn(batch):
    input_batch, answer_batch, mask_batch = [], [], []
    img_tmp,src_length_batch,name_batch = [], [], []

    for name_sample, img_sample, input, answer, mask in batch:

        name_batch.append(name_sample)
        img_tmp.append(img_sample)
        input_batch.append(input)
        answer_batch.append(answer)
        mask_batch.append(mask)



    max_len = max([len(vid) for vid in img_tmp])
    video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
    left_pad = 8
    right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
    max_len = max_len + left_pad + right_pad
    padded_video = [torch.cat(
        (
            vid[0][None].expand(left_pad, -1, -1, -1),
            vid,
            vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
        )
        , dim=0)
        for vid in img_tmp]
    
    img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))]
    
    for i in range(len(img_tmp)):
        src_length_batch.append(len(img_tmp[i]))
    src_length_batch = torch.tensor(src_length_batch)
    
    img_batch = torch.cat(img_tmp,0)

    new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
    new_src_lengths = new_src_lengths.long()
    mask_gen = []
    for i in new_src_lengths:
        tmp = torch.ones([i]) + 7
        mask_gen.append(tmp)
    mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
    img_padding_mask = (mask_gen != PAD_IDX).long()
    
    # with self.tokenizer.as_target_tokenizer():
    #     tgt_input = self.tokenizer(tgt_batch, return_tensors="pt",padding = True,  truncation=True)

    # input_batch = pad_sequence(input_batch, padding_value=-1, batch_first=True)
    # answer_batch = pad_sequence(answer_batch, padding_value=-1, batch_first=True)
    input_batch = torch.stack(input_batch)
    answer_batch = torch.stack(answer_batch)
    mask_batch = torch.stack(mask_batch)


    src_input = {}
    src_input['input_ids'] = img_batch
    src_input['attention_mask'] = img_padding_mask
    src_input['name_batch'] = name_batch

    src_input['src_length_batch'] = src_length_batch
    src_input['new_src_length_batch'] = new_src_lengths
    
    # if training_refurbish:
    #     masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
    #     with self.tokenizer.as_target_tokenizer():
    #         masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
    #     return src_input, tgt_input, masked_tgt_input
    return src_input, input_batch, answer_batch, mask_batch

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        assert self.len1 == self.len2, "Datasets should be of equal length for a 50-50 split."
    
    def __len__(self):
        return self.len1 + self.len2
    
    def __getitem__(self, idx):
        # Alternate between the two datasets
        if idx % 2 == 0:  # Even index: sample from dataset1
            return self.dataset1[idx // 2]
        else:             # Odd index: sample from dataset2
            return self.dataset2[idx // 2]



class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            root_text_path,
            qa_csv_path,
            tokenizer_path,
            data_config: dict|str,
            resize=256,
            input_size=224,
            batch_size=1, 
            num_workers=1):
        super().__init__()
        self.text_train = root_text_path + '.train'
        self.text_val = root_text_path + '.dev'
        self.text_test = root_text_path + '.test'

        self.qa_csv_path = qa_csv_path
        self.tokenizer_path = tokenizer_path

        if type(data_config) == str:
            with open(data_config, 'r') as file:
                self.data_config = yaml.safe_load(file)
        else:
            self.data_config = data_config

        self.resize = resize
        self.input_size = input_size

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str|None = None):
        
        if stage == 'fit' or stage is None:
            # tran and valdiation dataset
            train_slt_dataset = S2T_Dataset(path=self.text_train, tokenizer_path=self.tokenizer_path, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')
            train_sqa_dataset = SQA_Dataset(path=self.qa_csv_path, tokenizer_path=self.tokenizer_path, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')
            self.train_dataset = CombinedDataset(train_slt_dataset, train_sqa_dataset)

            self.val_dataset = S2T_Dataset(path=self.text_val, tokenizer_path=self.tokenizer_path, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='dev')

        if stage == 'test' or stage is None:
            # test dataset
            self.test_dataset = S2T_Dataset(path=self.text_test, tokenizer_path=self.tokenizer_path, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)


if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    PATH_ = '/mnt/fast/nobackup/users/sa04359'
    tokenizer_path = f'{PATH_}/llama_dir/tokenizer.model'
    qa_csv_path = 'src/sqa/data/clean-qa.csv'
    root_text_path = 'src/sqa/data/labels'
    phase = 'train'
    data_module = DataModule(
        root_text_path,
        qa_csv_path,
        tokenizer_path,
        data_config=config,
        batch_size=4,
    )

    # # sample = dataset[0]
    # # Create the DataLoader
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    data_module.setup()
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_dataloader = data_module.train_dataloader()
    # print(dataloader)

    # Example training loop
    for idx, (src_input, input_batch, answer_batch, mask_batch) in enumerate(train_dataloader):
        print(input_batch.shape)
        print(answer_batch.shape)
        print(mask_batch.shape)
        # print(input_batch[0,:])
        # print(answer_batch[0,:])
        break


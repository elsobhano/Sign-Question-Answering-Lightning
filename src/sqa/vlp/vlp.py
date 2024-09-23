import torch
import torch.nn as nn
from torch import Tensor
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence

import torchvision
from transformers import MBartForConditionalGeneration
import numpy as np

import humanize
from copy import deepcopy
from einops import repeat
from .definition import *
from typing import Dict
from sqa.logging import logger

# visual_encoder_path = './src/sqa/vlp/pretrain_models/mytran/'
# trasnformer_path = './src/sqa/vlp/pretrain_models/MBart_trimmed/'


def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format('resnet'))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src


class ImageCLIP(nn.Module):
    def __init__(self,
                visual_encoder_path,
                inplanes=1024, planes=1024, head_type='linear') :
        super(ImageCLIP, self).__init__()
        # self.config = config
        self.model =  FeatureExtracter() 
        
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(visual_encoder_path).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)
        
    def forward(self, src_input):
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        # print(x.shape)
        # exit()
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        # print(last_hidden_state.shape)
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output
    

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class TextCLIP(nn.Module):
    def __init__(self,
                trasnformer_path,
                inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(trasnformer_path).get_encoder() 

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits
    


class SLRCLIP(nn.Module):
    def __init__(self,
                trasnformer_path,
                visual_encoder_path,
                embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(trasnformer_path, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(visual_encoder_path, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        image_features = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth
    
class Video_Encoder(nn.Module):
    def __init__(self,
                trasnformer_path,
                visual_encoder_path,
                ckpt_path: str|None = None,
                dim_model: int = 1024,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        vlp_model = SLRCLIP(trasnformer_path, visual_encoder_path)
        self.dim_model = dim_model

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            ret =  vlp_model.load_state_dict(checkpoint['model'], strict=False)
            logger.info(f"Video encoder weights were loaded")
        
            
        
        self.model = deepcopy(vlp_model.model_images.model)
        self.video_encoder = deepcopy(vlp_model.model_images.trans_encoder)
        self.cls_token = deepcopy(vlp_model.model_images.cls_token)
        # self.video_encoder.lm_head = nn.Identity()

    def forward(self, src_input:Dict[str, torch.Tensor]):
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        # print(x.shape)
        # exit()
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.shape)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.video_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        return last_hidden_state
        


    

if __name__ == '__main__':

    print(f"Creating model:")
    # model = SLRCLIP().cuda()
    # model.to(device)
    dummy_input = torch.randn(128,3,224,224)
    src_length_batch = [64,64]
    src_length_batch = torch.tensor(src_length_batch)
    new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
    new_src_lengths = new_src_lengths.long()
    mask_gen = []
    for i in new_src_lengths:
        tmp = torch.ones([i]) + 7
        mask_gen.append(tmp)
    mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
    img_padding_mask = (mask_gen != PAD_IDX).long()
    src_input = {
        'input_ids': dummy_input,
        'src_length_batch': src_length_batch,
    }
    src_input['attention_mask'] = img_padding_mask
    
    ckpt_path = './best_checkpoint.pth'
    video_encoder = Video_Encoder(ckpt_path=ckpt_path).cuda()
    out = video_encoder(src_input)
    # checkpoint = torch.load(ckpt_path, map_location='cpu')
    # ret =  model.load_state_dict(checkpoint['model'], strict=False)
    # print('Missing keys: \n', '\n'.join(ret.missing_keys))
    # print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    # visual_model = deepcopy(model.model_images)
    # print(visual_model)
    # visual_model.lm_head = nn.Identity()
    # print(visual_model)
    print(video_encoder.dim_model)
    total_params = sum(p.numel() for p in video_encoder.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {humanize.intword(total_params)}')

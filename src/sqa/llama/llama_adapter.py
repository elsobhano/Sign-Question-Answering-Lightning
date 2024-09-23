import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn

import pytorch_lightning as pl

from timm.models.vision_transformer import Block

from sqa.llama import ModelArgs, Transformer
from sqa.llama.tokenizer import Tokenizer
from sqa.llama.utils import sample_top_p, _download
from sqa.llama.evaluate import compute_bleu

from sqa.vlp import Video_Encoder
from sqa.vlp import PAD_IDX
from torch.nn.utils.rnn import pad_sequence

import humanize

from sqa.logging import logger

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LLaMA_adapter(pl.LightningModule):

    def __init__(self, 
                llama_ckpt_dir, llama_tokenizer,
                max_seq_len=512, max_batch_size=1,
                encoder_type: str = 'vlp',
                encoder_path: str = 'src/sqa/vlp/best_checkpoint.pth',
                v_embed_dim=768, v_depth=8,
                v_num_heads=16, v_mlp_ratio=4.0,
                encoder_train=False,
                query_len=10, query_layer=31,
                w_bias=False, 
                w_lora=False, lora_rank=16,
                lora_train=False, 
                w_new_gate=False,
                batch_size=None,
                num_gpus=None,
                lr=None,
                blr=1e-3,
                weight_decay=None,
                phase="finetune"):
        super().__init__()

        logger.info(f"LLaMA Adapter Class initilaization in {phase} mode.")
        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        w_bias = phase == "finetune"
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc


        # 1. video encoder
        if encoder_type == 'vlp':
            logger.info(f"The video encoder model is being built ...")
            self.video_encoder = Video_Encoder(ckpt_path= encoder_path)
            logger.info(f"Video encoder model is  built")
            encoder_dim = self.video_encoder.dim_model

        

        # clip_dim = self.clip.visual.proj.shape[1]
        self.vlp_proj = nn.Linear(encoder_dim, v_embed_dim)
        self.vlp_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)
        self.pad_token_idx = self.tokenizer.pad_id
        self.eos_token_idx = self.tokenizer.eos_id
        self.bos_token_idx = self.tokenizer.bos_id
        logger.info(f"Tokenizer is built")

        # 5. llama
        model_args.w_bias = w_bias
        model_args.w_lora = w_lora
        model_args.lora_rank = lora_rank
        model_args.w_new_gate = w_new_gate
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        logger.info(f"LLaMA is built")
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
            logger.info(f"LLaMA weights are loaded")

        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # 7. training parameters
        self.encoder_train = encoder_train
        self.lora_train = lora_train

        self.phase = phase
        self.get_trainable_params(self.phase)

        # 8. learning rate
        self.lr = lr
        eff_batch_size = batch_size * num_gpus
        if self.lr is None:  # only base_lr is specified
            self.lr = blr * eff_batch_size / 256
        self.weight_decay = weight_decay
        


    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False
        logger.info(f"Freezing all the weights in model")


        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
            logger.info(f"Unfreezing some weights in  phase = {phase}")

        elif phase == 'pretrain':
            # train_param_name = ['video_encoder','gate', 'vlp_proj', 'vlp_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            train_param_name = ['gate', 'vlp_proj', 'vlp_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            if self.encoder_train:
                train_param_name.append('video_encoder')
            if self.lora_train:
                train_param_name.append('lora')
                
            
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True
            logger.info(f"Unfreezing some weights in  phase = {phase}")
                        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
        
    
    def forward_visual(self, videos):
        # This module prepares visual queries for llm
        visual_feats = self.video_encoder(videos)
        bsz = visual_feats.shape[0]
        # print(visual_feats.shape)
        vlp_feats = self.vlp_proj_norm(self.vlp_proj(visual_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(bsz, 1, 1)
        # print(visual_query.shape, clip_feats.shape)
        visual_query = torch.cat([visual_query, vlp_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward_visual_clip(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, 
                tokens, 
                videos,
                input_mask=None):
        # print(tokens.shape, labels.shape)
        
        visual_query = self.forward_visual(videos)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        
        if input_mask == None:
            mask = torch.tril(torch.ones((seqlen, seqlen), device=h.device)).unsqueeze(0).unsqueeze(1)        
        
        else:
            padding_mask = input_mask.unsqueeze(1).unsqueeze(2).float()
            causal_mask = torch.tril(torch.ones((seqlen, seqlen), device=h.device)).unsqueeze(0).unsqueeze(1)
            mask = padding_mask * causal_mask

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]: # the last (self.query_layer)-th of llamma
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)

        return output
    
    
    def training_step(self, batch, batch_idx):
        src_input, input_batch, answer_batch, mask_batch = batch
        logits = self.forward(input_batch, src_input, mask_batch)
        loss = self.loss_function(logits, answer_batch)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_input, input_batch, answer_batch, mask_batch = batch
        logits = self.forward(input_batch, src_input, mask_batch)
        loss = self.loss_function(logits, answer_batch)
        bleu_score = self.calculate_belu(logits, answer_batch)

        self.log_dict({
            "val_loss": loss,
            "val_bleu": bleu_score}, on_step=False, on_epoch=True, prog_bar=True)
        
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        src_input, input_batch, answer_batch, mask_batch = batch
        logits = self.forward(input_batch, src_input, mask_batch)
        loss = self.loss_function(logits, answer_batch)
        bleu_score = self.calculate_belu(logits, answer_batch)
        
        self.log("test_loss", loss)
        self.log("test_bleu", bleu_score)
        return loss
    
    def loss_function(self, output, labels):
        assert self.llama.vocab_size == 32000
        c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss
    
    def remove_special_tokens(self, sentence_tokens):
        """Removes padding tokens and truncates at EOS."""
        
        sentence_tokens = [token for token in sentence_tokens if token not in [self.bos_token_idx, self.pad_token_idx, 0]]
        if self.eos_token_idx in sentence_tokens:
            sentence_tokens = sentence_tokens[:sentence_tokens.index(self.eos_token_idx)]
        return sentence_tokens

    def calculate_belu(self, logits, tgt):
        
        pred_tokens = logits.argmax(dim=-1)
        bleu_scores = []
        
        for i in range(pred_tokens.size(0)):  # Loop over the batch
            pred_seq = pred_tokens[i].tolist()  # Get the predicted token indices for the i-th sample
            tgt_seq = tgt[i].tolist()  # Get the target token indices for the i-th sample

             # Remove padding and EOS tokens from both sequences
            pred_seq = self.remove_special_tokens(pred_seq)
            tgt_seq = self.remove_special_tokens(tgt_seq)

            # Wrap the target sequence in a list (expects multiple references)
            reference_corpus = [[tgt_seq]]
            translation_corpus = [pred_seq]

            # Calculate BLEU-4 for this sample
            bleu_score, precisions, bp, ratio, translation_length, reference_length = compute_bleu(
                reference_corpus, translation_corpus, max_order=4, smooth=True)

            bleu_scores.append(bleu_score)

        # Calculate the mean BLEU score for the batch
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        
        return avg_bleu
    
    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Ignore frozen parameters
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}
        ]
    
    def configure_optimizers(self):
        """Define the optimizer and pass custom weight decay parameters."""
        param_groups = self.add_weight_decay(self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        return optimizer

        
    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded



_MODELS = {
    "BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    "LORA-BIAS-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth",
    "CAPTION-7B": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth",
    "LORA-BIAS-7B-v21": "https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, 
        llama_dir, 
        llama_type="7B", 
        max_seq_len=512,
        encoder_path='src/sqa/vlp/best_checkpoint.pth',
        v_embed_dim=768, v_depth=8,
        v_num_heads=16, v_mlp_ratio=4.0,
        encoder_train=False,
        lora_train=False,
        query_len=10, query_layer=31,
        batch_size=None,
        num_gpus=None,
        lr=None,
        blr=1e-3,
        weight_decay=None,
        device="cuda", 
        download_root='ckpts', 
        phase="pretrain"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}"), None

    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    # llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    logger.info(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})
    print(model_cfg)

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=max_seq_len, max_batch_size=1,
        encoder_type='vlp',
        encoder_path=encoder_path,
        v_embed_dim=v_embed_dim, v_depth=v_depth,
        v_num_heads=v_num_heads, v_mlp_ratio=v_mlp_ratio,
        encoder_train=encoder_train,
        query_len=query_len, query_layer=query_layer,
        w_bias=model_cfg.get('w_bias', False), 
        w_lora=model_cfg.get('w_lora', False), 
        lora_rank=model_cfg.get('lora_rank', 16),
        lora_train=lora_train,
        w_new_gate=model_cfg.get('w_lora', False), # for compatibility
        batch_size=batch_size,
        num_gpus=num_gpus,
        lr=lr,
        blr=blr,
        weight_decay=weight_decay,
        phase=phase)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of trainable parameters of llama adapter: {humanize.intword(total_params)}')


    # load_result = model.load_state_dict(ckpt['model'], strict=False)

    # assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device)#, model.clip_transform

if __name__ == "__main__":
    model = load(name="LORA-BIAS-7B-v21",llama_dir='./src/sqa/llama/llama_dir', batch_size=2, num_gpus=1)
    print(model.named_parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {humanize.intword(total_params)}')

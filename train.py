import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

import numpy as np

from sqa.llama.llama_adapter import load
from sqa.dataset.slt_dataset import DataModule

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.sqa.utils.common import load_yaml

import wandb
API_TOKEN_WANDB = "1af8cc2a4ed95f2ba66c31d193caf3dd61c3a41f"
wandb.login(key=API_TOKEN_WANDB)

import argparse
from pathlib import Path
from datetime import datetime

torch.set_float32_matmul_precision("medium") # to make lightning happy


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_name', default='LORA-BIAS-7B-v21', type=str,
                        help='Name of pretrained LLaMA model')
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_dir', default='src/sqa/llama/llama_dir', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--max_words', default=128, type=int,
                        help='max number of input words')

    # Video encoder parameters
    parser.add_argument('--encoder_train', default=False, type=bool,
                        help='Train video encoder or not')
    parser.add_argument('--encoder_path', default='src/sqa/vlp/best_checkpoint.pth', type=str,
                        help='best weight for video encoder')
    
    parser.add_argument('--v_embed_dim', default=768, type=int,
                        )
    parser.add_argument('--v_depth', default=8, type=int,
                        ) #
    parser.add_argument('--v_num_heads', default=16, type=int,
                        )
    parser.add_argument('--v_mlp_ratio', default=4.0, type=float,
                        )

    # Adapter parameters
    parser.add_argument('--query_len', default=10, type=int,
                        )
    parser.add_argument('--query_layer', default=31, type=int,
                        )
    parser.add_argument('--lora_train', default=False, type=bool,
                        help='Train lora or not')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='config/config.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--tknzr_path', default='src/sqa/llama/llama_dir/tokenizer.model', type=str,
                        help='path to tokenizer')
    parser.add_argument('--text_path', default='src/sqa/data/labels', type=str,
                        help='path to translated data')
    parser.add_argument('--qa_path', default='src/sqa/data/clean-qa.csv', type=str,
                        help='path to question-answer path')
    
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--input_size', default=224, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--patience', default=5, type=int,
                        help='early stop patience')

    return parser

def main(args):

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # set logger
    wandb_logger = WandbLogger(save_dir=args.log_dir, project="SQA-LLaMA", name="Test")
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dirpath = f'{args.output_dir}/run_{current_time}'
    
    # set callbacks
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    monitor="val_bleu",
    mode="max",
    dirpath=dirpath,
    filename="best-{epoch:03d}-{val_bleu:.3f}",
    )
    early_stop = EarlyStopping("val_bleu", patience=args.patience, mode="max", verbose=True)
    callbacks = [checkpoint_callback, early_stop]


    num_gpus = 1
    model = load(
        name=args.llama_name, 
        llama_dir=args.llama_dir, 
        llama_type=args.llama_type, 
        max_seq_len=args.max_words,
        encoder_path=args.encoder_path,
        v_embed_dim=args.v_embed_dim, v_depth=args.v_depth,
        v_num_heads=args.v_num_heads, v_mlp_ratio=args.v_mlp_ratio,
        encoder_train=args.encoder_train,
        lora_train=args.lora_train, 
        query_len=args.query_len, query_layer=args.query_layer,
        batch_size=args.batch_size,
        num_gpus=num_gpus,
        lr=args.lr,
        blr=args.blr,
        weight_decay=args.weight_decay,
        device=args.device,
        phase="pretrain",
        download_root='ckpts', 
    )

    data_module = DataModule(
        args.text_path,
        args.qa_path,
        args.tknzr_path,
        args.data_config,
        resize=args.resize,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    compute = load_yaml(args.data_config)['compute']

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=compute["accelerator"],
        devices=compute["devices"],
        min_epochs=1,
        max_epochs=args.epochs,
        precision=compute["precision"],
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)
    # pass

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

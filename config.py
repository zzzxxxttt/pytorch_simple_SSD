import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/data1/zhuxt/VOCdevkit')
parser.add_argument('--eval_data_dir', type=str, default='/data1/zhuxt/VOCdevkit_test/')
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='SSD_07_12_glasso_0.01')

parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrain_dir', type=str, default='./model')
parser.add_argument('--base_network', type=str, default='vgg16')

parser.add_argument('--neg_pos_ratio', type=float, default=3.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=1e-4)

parser.add_argument('--filter_wise', action='store_true', default=False)
parser.add_argument('--channel_wise', action='store_true', default=False)

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--epochs_per_eval', type=int, default=10)

parser.add_argument('--workers', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.model_dir = os.path.join(cfg.root_dir, 'model', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.base_dir = os.path.join(cfg.root_dir, 'model', cfg.base_network)
os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)

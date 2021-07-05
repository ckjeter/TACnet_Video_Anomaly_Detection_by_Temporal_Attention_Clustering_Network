import argparse
import os
root = "/data/dataset/UCF_Crime"
segment_length = 16
segment_count = 32
fix_length = True 
if fix_length:
    token = 'L' + str(segment_count)
else:
    token = 'C' + str(segment_length)
train_path = os.path.join(root, 'features', token, 'train')
test_path = os.path.join(root, 'features', token, 'test')
#loss_parameter = [0.7, 0, 0.5, 0.5, 0.001, 0.001]
loss_parameter = [1, 0, 0.5, 0.3, 0.05, 0.05]
#loss_parameter = [0.5, 0.01, 0.1, 0.3, 0.01, 0.001]
def parse_args():
    parser = argparse.ArgumentParser(description="fuck")
    parser.add_argument('--gpus', type=str, default='0', help="multi GPUs?")
    parser.add_argument('--target', type=str, default='frame', help="multi GPUs?")
    parser.add_argument('--train_path', type=str, default=train_path)
    parser.add_argument('--test_path', type=str, default=test_path)
    parser.add_argument('--attention_type', type=str, default='normal')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--name', type=str, default='figure')
    parser.add_argument('--p_graph', action="store_true", help="performance_graph")
    parser.add_argument('--c_graph', action="store_true", help="cluster_graph")
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--savelog', action="store_true")
    parser.add_argument('--load_backbone', action="store_true")
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--drawmask', action="store_true")
    parser.add_argument('--drawattn', action="store_true")
    parser.add_argument('--use_saliency', type=float, default=9)
    return parser.parse_args()

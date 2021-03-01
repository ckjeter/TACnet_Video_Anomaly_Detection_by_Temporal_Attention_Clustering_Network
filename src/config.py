import argparse
import os
root_UCFCrime = "/data/dataset/UCF_Crime"
root_SH = "/data/dataset/ShanghaiTech"
#loss_parameter = [0.7, 0, 0.5, 0.5, 0.001, 0.001]
loss_parameter = [1, 0.5, 0.3, 0.05, 0.05]
#loss_parameter = [0.5, 0.01, 0.1, 0.3, 0.01, 0.001]
def parse_args():
    parser = argparse.ArgumentParser(description="TACnet")
    parser.add_argument('--gpus', type=str, default='0', help="multi GPUs?")
    parser.add_argument('--attention_type', type=str, default='gate')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--name', type=str, default='figure')
    parser.add_argument('--p_graph', action="store_true", help="performance_graph")
    parser.add_argument('--c_graph', action="store_true", help="cluster_graph")
    parser.add_argument('--savelog', action="store_true")
    parser.add_argument('--load_pretrain', action="store_true")
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--attn_graph', action="store_true")
    parser.add_argument('--dataset', type=str, required=True, help="UCF/SH")
    return parser.parse_args()

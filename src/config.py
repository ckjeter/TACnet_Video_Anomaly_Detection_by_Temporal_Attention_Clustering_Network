import argparse
root = "/data/dataset/UCF_Crime"
segment_length = 16
def parse_args():
    parser = argparse.ArgumentParser(description="fuck")
    parser.add_argument('--gpus', type=str, default='0', help="multi GPUs?")
    parser.add_argument('--train_path', type=str, default='features/train')
    parser.add_argument('--test_path', type=str, default='features/test')
    parser.add_argument('--attention_type', type=str, default='normal')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--name', type=str, default='figure')
    parser.add_argument('--draw', action="store_true")
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--savelog', action="store_true")
    return parser.parse_args()

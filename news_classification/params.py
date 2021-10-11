from argparse import ArgumentParser


def parse_args():
    
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prop_split", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device_no", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--max_epoch", type=int)
    return parser.parse_args()
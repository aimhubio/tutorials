import time
import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from classifiers import TextClassificationModel
from datas import collate_batch
from params import parse_args


def build_vocab(data_path):
    train_iter = AG_NEWS(root=data_path, split='train')
    # build vocab based on train corpus
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    return text_pipeline, label_pipeline, len(vocab)


def trainer(args, text_pipeline, label_pipeline, vocab_size):
    # Generate data loader
    train_iter, test_iter = AG_NEWS(root=args.data_path, split=('train', 'test'))
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * args.prop_split)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_loader = DataLoader(split_train_, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline, args.device_no))
    valid_loader = DataLoader(split_valid_, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline, args.device_no))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_batch(b, text_pipeline, label_pipeline, args.device_no))

    # Initiate an model
    temp_iter = AG_NEWS(root=args.data_path, split='train')
    num_class = len(set([label for (label, text) in temp_iter]))
    text_classifier = TextClassificationModel(vocab_size=vocab_size, embed_dim=args.embed_dim, num_class=num_class)
    if args.device_no > -1:
        text_classifier = text_classifier.cuda(args.device_no)

    optimizer = torch.optim.SGD(text_classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    def train(data_loader, epoch):
        text_classifier.train()
        total_acc, total_count = 0, 9
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(data_loader):
            optimizer.zero_grad()
            predicted_label = text_classifier(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_classifier.parameters(), 0.1)
            optimizer.step()

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % args.log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(data_loader),
                                                  total_acc / total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(data_loader):
        text_classifier.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(data_loader):
                predicted_label = text_classifier(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    total_accuracy = None
    for epoch_no in range(1, args.max_epoch + 1):
        epoch_start_time = time.time()

        train(data_loader=train_loader, epoch=epoch_no)

        accuracy_val = evaluate(data_loader=valid_loader)
        if total_accuracy is not None and total_accuracy > accuracy_val:
            scheduler.step()
        else:
            total_accuracy = accuracy_val

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch_no,
                                               time.time() - epoch_start_time,
                                               accuracy_val))
        print('-' * 59)

    # check the results on test set.
    print('Checking the results of test dataset.')
    accu_test = evaluate(test_loader)
    print('test accuracy {:8.3f}'.format(accu_test))


if __name__ == '__main__':

    args = parse_args()
    trainer(args, *build_vocab(args.data_path))

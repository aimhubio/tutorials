import torch


def collate_batch(batch, text_pipeline, label_pipeline, device_no=-1):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    if device_no > -1:
        return label_list.cuda(device_no), text_list.cuda(device_no), offsets.cuda(device_no)
    else:
        return label_list, text_list, offsets

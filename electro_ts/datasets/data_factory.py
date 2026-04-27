from electro_ts.datasets.data_loader import ElectroDataset
from torch.utils.data import DataLoader


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_set = ElectroDataset(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len, args.enc_in],
        time_cutoff=args.time_cutoff,
        test_idx=args.test_idx,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False)
    return data_set, data_loader

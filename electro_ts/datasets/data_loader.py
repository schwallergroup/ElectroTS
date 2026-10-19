import os
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def get_ymean_time(arr, xmin, xmax):
    x = arr[:, 0]
    y = arr[:, 1]
    mask = (x > xmin) & (x < xmax)
    return y[mask].mean()


class ElectroDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 scale=True, time_cutoff=400, test_idx=0):
        if size is None:
            self.seq_len = 60
            self.label_len = 10
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']

        self.scale = scale
        self.partition = flag

        self.root_path = root_path
        self.time_cutoff = time_cutoff
        self.test_idx = test_idx
        self.__read_data__()

    def __read_data__(self):
        data_dict = {}
        _path = os.path.join(self.root_path, 'all')
        txt_list = sorted(f for f in os.listdir(_path) if f.endswith('.txt'))
        arr_list = [
            np.loadtxt(os.path.join(_path, _txt))
            for _txt in txt_list
        ]

        new_arr_list = []
        for arr in arr_list:
            tstart = arr[0, 0]
            y_ref = get_ymean_time(arr, tstart, tstart + 1)
            arr[:, 1] = arr[:, 1] - y_ref
            new_arr_list.append(arr)

        above_arr = []
        below_arr = []
        for i, arr in enumerate(new_arr_list):
            tmax = arr[:, 0].max()
            if tmax > self.time_cutoff:
                above_arr.append(arr)
            else:
                below_arr.append(arr)
        train_arr = below_arr + above_arr[:self.test_idx] + above_arr[self.test_idx+1:]
        valid_arr = train_arr
        test_arr = [above_arr[self.test_idx]]
        arr_dict = {
            'train': train_arr,
            'val': valid_arr,
            'test': test_arr,
        }

        for partition in ['train', 'val', 'test']:
            data_x, stamp, map = [], [], []
            for i, arr in enumerate(arr_dict[partition]):
                data_x.append(arr[:, 1:])
                stamp.append(arr[:, 0:1])

                data_num = len(arr) - self.seq_len - self.pred_len + 1
                for j in range(data_num):
                    map.append((i, j))
            data_dict[partition] = {'x': data_x, 'stamp': stamp, 'map': map}

        self.scaler = StandardScaler()
        if self.scale:
            train_data = np.concatenate(data_dict['train']['x'], axis=0)
            self.scaler.fit(train_data)
            for partition in ['train', 'val', 'test']:
                for i in range(len(data_dict[partition]['x'])):
                    data_dict[partition]['x'][i] = self.scaler.transform(data_dict[partition]['x'][i])

        self.data_dict = data_dict

    def __getitem__(self, index):
        idx1, idx2 = self.data_dict[self.partition]['map'][index]
        data_x = self.data_dict[self.partition]['x'][idx1]
        stamp = self.data_dict[self.partition]['stamp'][idx1]
        s_begin = idx2
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_x[r_begin:r_end]
        seq_x_mark = stamp[s_begin:s_end]
        seq_y_mark = stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_dict[self.partition]['map'])

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

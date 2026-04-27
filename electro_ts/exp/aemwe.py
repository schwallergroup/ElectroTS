import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from electro_ts.datasets.data_factory import data_provider
from electro_ts.exp.exp_basic import Exp_Basic
from electro_ts.utils.eval import compute_errors, decay_rate, lifetime
from electro_ts.utils.tools import AverageMeter, EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

# Voltage channel slice used in all loss computations
_V_START, _V_END = 0, 1


def _get_ymean_in_window(arr, t_start, t_end):
    mask = (arr[:, 0] > t_start) & (arr[:, 0] < t_end)
    return arr[:, 1][mask].mean()


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.restart_path is not None:
            model.load_state_dict(torch.load(self.args.restart_path))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, _V_START:_V_END]
                batch_y = batch_y[:, -self.args.pred_len:, _V_START:_V_END].to(self.device)
                total_loss.update(criterion(outputs, batch_y).item(), batch_x.size(0))

        self.model.train()
        return total_loss.avg

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            iter_count = 0
            train_loss = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, _V_START:_V_END]
                batch_y = batch_y[:, -self.args.pred_len:, _V_START:_V_END]
                loss = criterion(outputs, batch_y)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if (i + 1) % 20 == 0:
                    train_loss.append(loss.item())
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}')
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            train_loss_avg = np.mean(train_loss) if train_loss else float('nan')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                f'Epoch {epoch + 1} ({time.time() - epoch_time:.1f}s) | '
                f'Train: {train_loss_avg:.7f}  Val: {vali_loss:.7f}  Test: {test_loss:.7f}'
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        torch.save(self.model.state_dict(), os.path.join(path, 'last.pth'))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data('train')
        _, test_loader = self._get_data('test')
        train_mean = torch.from_numpy(train_data.scaler.mean_).float().to(self.device)
        train_std = torch.from_numpy(train_data.scaler.scale_).float().to(self.device)
        print(f'Train mean: {train_mean}  Train std: {train_std}')

        if test:
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        mse_meter = AverageMeter()
        mae_meter = AverageMeter()
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mae_time = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs * train_std + train_mean
                batch_y = batch_y * train_std + train_mean

                outputs = outputs[:, -self.args.pred_len:, _V_START:_V_END]
                batch_y = batch_y[:, -self.args.pred_len:, _V_START:_V_END]

                mse_meter.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae_meter.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))
                mae_time.extend(torch.abs(outputs - batch_y).mean((1, 2)).cpu().numpy().tolist())
                pred_list.append({
                    'x_time': batch_x_mark.cpu().numpy(),
                    'x': (batch_x * train_std + train_mean).cpu().numpy(),
                    'y_time': batch_y_mark[:, -self.args.pred_len:, :].cpu().numpy(),
                    'y': batch_y.cpu().numpy(),
                    'y_pred': outputs.cpu().numpy(),
                })

        print(f'MSE: {mse_meter.avg:.6f}  MAE: {mae_meter.avg:.6f}')
        # np.savetxt('mae_time.txt', np.array(mae_time), fmt='%.6f')
        with open('pred_list.pkl', 'wb') as f:
            pickle.dump(pred_list, f)

    # ------------------------------------------------------------------
    # Autoregressive prediction
    # ------------------------------------------------------------------

    def predict(self, setting, txt_path):
        train_data, _ = self._get_data('train')
        train_mean = torch.from_numpy(train_data.scaler.mean_).float().to(self.device)
        train_std = torch.from_numpy(train_data.scaler.scale_).float().to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
        )

        true_data = np.loadtxt(txt_path)
        t0 = true_data[0, 0]
        y_ref = _get_ymean_in_window(true_data, t0, t0 + 1)

        pred_time_1d = true_data[:, 0]
        tstep = pred_time_1d[1] - pred_time_1d[0]
        total_len = len(pred_time_1d) + self.args.pred_len
        pred_time = np.arange(t0, t0 + total_len * tstep, tstep).reshape(-1, 1)

        pred_y = true_data[:, 1:][:self.args.seq_len] - y_ref
        pred_steps = int((len(pred_time) - self.args.seq_len) / self.args.pred_len)

        self.model.eval()
        with torch.no_grad():
            for i in range(pred_steps):
                idx = self.args.seq_len + self.args.pred_len * i
                batch_x = torch.tensor([pred_y[-self.args.seq_len:]]).float().to(self.device)
                batch_x = (batch_x - train_mean) / train_std
                batch_x_mark = torch.tensor(
                    [pred_time[idx - self.args.seq_len: idx]]
                ).float().to(self.device)
                batch_y_mark = torch.tensor(
                    [pred_time[idx - self.args.label_len: idx + self.args.pred_len]]
                ).float().to(self.device)

                dec_inp = torch.zeros((1, self.args.pred_len, batch_x.size(2))).float().to(self.device)
                dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1).float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                step_pred = (outputs * train_std + train_mean)[:, -self.args.pred_len:, :]
                pred_y = np.concatenate((pred_y, step_pred[0].cpu().numpy()), axis=0)

        pred_y = pred_y + y_ref
        pred_arr_full = np.hstack((pred_time[:len(pred_y)], pred_y))
        pred_arr_obs = np.hstack((pred_time[:len(true_data)], pred_y[:len(true_data)]))

        true_t = true_data[:, 0]
        true_v = true_data[:, 1]
        pred_v = pred_y.flatten()[:len(true_data)]

        # Save predictions to CSV
        ckpt_dir = os.path.join('./checkpoints/' + setting)
        pd.DataFrame({'time': true_t, 'true_y': true_v, 'pred_y': pred_v}).to_csv(
            os.path.join(ckpt_dir, f'{os.path.basename(txt_path)}_predict.csv'), index=False
        )

        # Compute metrics
        true_decay = decay_rate(true_data)
        pred_decay = decay_rate(pred_arr_obs)
        decay_ae, decay_ape = compute_errors(true_decay, pred_decay)

        true_lt = lifetime(true_data, ratio=0.05)
        pred_lt = lifetime(pred_arr_full, ratio=0.05)
        if true_lt is None:
            lt_ae, lt_ape = None, None
        else:
            if pred_lt is None:
                pred_lt = pred_time.max()
            lt_ae, lt_ape = compute_errors(true_lt, pred_lt)

        vol_ae = np.abs(true_v[self.args.seq_len:] - pred_v[self.args.seq_len:])
        vol_mae = float(np.mean(vol_ae))
        vol_std = float(np.std(vol_ae))
        vol_ae_max = float(np.max(vol_ae))

        # Print summary
        print(f'--- {txt_path} ---')
        print(f'Decay rate  true={true_decay:.4f}  pred={pred_decay:.4f}  AE={decay_ae:.4f}  APE={decay_ape:.4f}%')
        if lt_ae is None:
            print(f'Lifetime    true={true_lt}  pred={pred_lt}  AE={lt_ae}  APE={lt_ape}')
        else:
            print(f'Lifetime    true={true_lt:.4f}  pred={pred_lt:.4f}  AE={lt_ae:.4f}  APE={lt_ape:.4f}%')
        print(f'Voltage     MAE={vol_mae:.4f}  STD={vol_std:.4f}  MAX={vol_ae_max:.4f}')

        # Plot
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(true_t, true_v, label='True')
        ax.plot(true_t[self.args.seq_len:], pred_v[self.args.seq_len:], label='Predict')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Cell Voltage (V)')
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f'{os.path.basename(txt_path)}.png'), dpi=300)
        plt.close(fig)

        return {
            'decay_ae': decay_ae,
            'decay_ape': decay_ape,
            'lt_ae': lt_ae,
            'lt_ape': lt_ape,
            'vol_mae': vol_mae,
            'vol_std': vol_std,
            'vol_ae_max': vol_ae_max,
        }

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import json
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 
    def _unpack_batch(self, batch):
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, observation_mask = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, observation_mask
        raise ValueError(f"Expected batch with 4 or 5 items, got {len(batch)} items.")

    def _call_model(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _forecast_feature_slice(self, tensor):
        f_dim = -1 if self.args.features == 'MS' else 0
        return tensor[:, -self.args.pred_len:, f_dim:]

    def _resolve_eval_observation_mask(self, observation_mask):
        if self.args.eval_mask_mode == 'all':
            return None
        if self.args.eval_mask_mode == 'observed':
            if observation_mask is None:
                raise ValueError("eval_mask_mode='observed' requires observation_mask in every evaluation batch.")
            return observation_mask
        if self.args.eval_mask_mode == 'auto':
            return observation_mask
        raise ValueError(f"Unsupported eval_mask_mode: {self.args.eval_mask_mode}")

    def _prepare_eval_mask(self, observation_mask, target_tensor):
        if observation_mask is None:
            return None, None, None

        mask = observation_mask.to(self.device)
        if mask.dtype != torch.bool:
            mask = mask != 0

        try:
            broadcast_mask = torch.broadcast_to(mask, target_tensor.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Observation mask with shape {tuple(mask.shape)} cannot broadcast to target shape {tuple(target_tensor.shape)}."
            ) from exc

        observed_time_count = int(mask.any(dim=-1).sum().item()) if mask.ndim >= 3 else int(mask.sum().item())
        evaluated_value_count = int(broadcast_mask.sum().item())
        if evaluated_value_count == 0:
            raise ValueError("Observation mask has zero valid evaluation points.")

        return broadcast_mask, observed_time_count, evaluated_value_count

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        squared_error_sum = 0.0
        valid_value_count = 0
        using_mask_eval = None
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, observation_mask = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._call_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = self._forecast_feature_slice(outputs)
                batch_y = self._forecast_feature_slice(batch_y)

                pred = outputs.detach()
                true = batch_y.detach()

                eval_mask = self._resolve_eval_observation_mask(observation_mask)
                batch_uses_mask = eval_mask is not None
                if using_mask_eval is None:
                    using_mask_eval = batch_uses_mask
                elif using_mask_eval != batch_uses_mask:
                    raise ValueError("Mixed masked and unmasked batches are not supported in a single validation loader.")

                if not batch_uses_mask:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
                    continue

                eval_mask, _, batch_value_count = self._prepare_eval_mask(eval_mask, pred)
                squared_error_sum += torch.sum((pred - true)[eval_mask] ** 2).item()
                valid_value_count += batch_value_count

        if using_mask_eval:
            if valid_value_count == 0:
                raise ValueError("Observation mask has zero valid evaluation points across the validation set.")
            total_loss = squared_error_sum / valid_value_count
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._call_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                        outputs = self._forecast_feature_slice(outputs)
                        batch_y = self._forecast_feature_slice(batch_y)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self._call_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    outputs = self._forecast_feature_slice(outputs)
                    batch_y = self._forecast_feature_slice(batch_y)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        raw_observed_masks = []
        eval_observed_masks = []
        using_mask_eval = None
        observed_time_count = None
        evaluated_value_count = None
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, observation_mask = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._call_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = self._forecast_feature_slice(outputs)
                batch_y = self._forecast_feature_slice(batch_y)

                pred = outputs
                true = batch_y

                eval_mask = self._resolve_eval_observation_mask(observation_mask)
                batch_uses_mask = eval_mask is not None
                if using_mask_eval is None:
                    using_mask_eval = batch_uses_mask
                elif using_mask_eval != batch_uses_mask:
                    raise ValueError("Mixed masked and unmasked batches are not supported in a single test loader.")

                if observation_mask is not None:
                    raw_observed_masks.append(observation_mask.detach().cpu().numpy())

                if batch_uses_mask:
                    eval_mask, batch_observed_time_count, batch_evaluated_value_count = self._prepare_eval_mask(
                        eval_mask, torch.from_numpy(pred).to(self.device)
                    )
                    observed_time_count = (observed_time_count or 0) + batch_observed_time_count
                    evaluated_value_count = (evaluated_value_count or 0) + batch_evaluated_value_count
                    eval_observed_masks.append(eval_mask.detach().cpu().numpy())

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        metric_mask = None
        evaluation_scope = 'all-points'
        if using_mask_eval:
            metric_mask = np.concatenate(eval_observed_masks, axis=0)
            evaluation_scope = 'observed-only'

        mae, mse, rmse, mape, mspe = metric(preds, trues, mask=metric_mask)
        print('evaluation_scope:{}, mse:{}, mae:{}, dtw:{}'.format(evaluation_scope, mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('evaluation_scope:{}, mse:{}, mae:{}, dtw:{}'.format(evaluation_scope, mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if raw_observed_masks:
            observed_masks = np.concatenate(raw_observed_masks, axis=0)
            np.save(folder_path + 'observed_mask.npy', observed_masks)
            if observed_time_count is None:
                observed_time_count = int(observed_masks.astype(bool).any(axis=-1).sum())
            if evaluated_value_count is None:
                evaluated_value_count = int(preds.size)

            metrics_real_only = {
                'evaluation_scope': evaluation_scope,
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe),
                'observed_time_count': int(observed_time_count),
                'evaluated_value_count': int(evaluated_value_count),
            }
            with open(folder_path + 'metrics_real_only.json', 'w') as handle:
                json.dump(metrics_real_only, handle, indent=2, sort_keys=True)

        return

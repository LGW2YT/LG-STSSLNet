import os
import sys
import torch
from shutil import copyfile
from lib.train_tool import EarlyStop, time_decorator
from lib.metrics import mask_evaluation_np
from lib.utils import contrast_loss
import lib.metrics
#from utils.metrics import get_mae, get_mape, get_rmse ##居然离谱


class Trainer:
    @staticmethod
    def get_eval_result(y_pred, y_true, risk_mask):
        module = sys.modules['lib.metrics']
        eval_results = []

        eval_func1 = getattr(module, 'mask_rmse_np')
        eval_func2 = getattr(module, 'Recall')
        eval_func3 = getattr(module, 'MAP')
        eval_results.append(eval_func1(y_true, y_pred,  risk_mask, 0))
        eval_results.append(eval_func2(y_true, y_pred, risk_mask))
        eval_results.append(eval_func3(y_true, y_pred, risk_mask))
        return eval_results

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def model_loss_func(self, y_pred, y_true, *args):
        return 0


class TFTrainer(Trainer):
    def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, patience):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epoch_num = max_epoch_num
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir = os.path.dirname(model_save_path)
        self.early_stop = EarlyStop(patience, min_is_best=True)
        self.device = next(self.model.parameters()).device


        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    @time_decorator
    def train(self, train_data_loader, eval_data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        tmp_state_save_path = os.path.join(self.model_save_dir, 'temp.pkl')
        min_loss = torch.finfo(torch.float32).max


        for epoch in range(1, self.max_epoch_num + 1):
            # train one epoch
            self.train_one_epoch(train_data_loader)

            # evaluate
            print('Epoch {}'.format(epoch), end='  ')
            eval_loss, _, _ = self.evaluate(eval_data_loader)
            print('Loss: {}'.format(eval_loss))
            # Criteria for early stopping
            if self.early_stop.reach_stop_criteria(eval_loss):
                break

            # save model state when meeting minimum loss
            # save to a temporary path first to avoid overwriting original state.
            if eval_loss < min_loss:
                print(' save temporary model state because of meeting minimum loss')
                torch.save(self.model.state_dict(), tmp_state_save_path)
                #new_adj_mx = self.model(self.best_adj_mx).detach()
                #print(new_adj_mx[:5,:5])
                min_loss = eval_loss



            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.early_stop.reset()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)

    @torch.no_grad()
    def evaluate(self, data_loader, adj_mx, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        self.model.eval()

        y_true, y_pred, loss,  = [], [], 0,
        for feature, target_time, gragh_feature, label in data_loader:
            feature, target_time, gragh_feature, label = feature.to(self.device), target_time.to(
                self.device), gragh_feature.to(self.device), label.to(self.device)
            pred, encode = self.model(feature, target_time, gragh_feature, adj_mx, self.grid_node_map)
            pred = pred.detach()
            encode = encode.detach()
            loss += self.model_loss_func(pred, label, encode, target_time, adj_mx).item()
            y_true.append(label)
            y_pred.append(pred)
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0).cpu().numpy())
        y_pred = self.scaler.inverse_transform(torch.cat(y_pred, dim=0).cpu().numpy())

        eval_results = self.get_eval_result(y_pred, y_true, self.risk_mask)
        for metric_name, eval_ret in zip(metrics, eval_results):
            print('{}:  {:.4f}'.format(metric_name.upper(), eval_ret), end='  ')#数据还原后计算的结果
        print()

        return loss / len(data_loader), y_true, y_pred #数据归一化的计算结果

    def print_test_result(self, y_pred, y_true, scaler, risk_mask):
        inverse_trans_pre = y_pred
        inverse_trans_label = y_true
        for i in range(y_true.shape[1]):
            results = mask_evaluation_np(inverse_trans_label[:, i], inverse_trans_pre[:, i], risk_mask, 0)
            print('Horizon {}'.format(i + 1), end='  ')
            metrics = ['RMSE', 'RECALL', 'MAP']
            for j in range(3):
                print('{}:  {:.4f}'.format(metrics[j], results[j]), end='  ')
            print()
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from .base import TFTrainer
from adj_mx import get_adj
import pickle as pkl
from lib.train_tool import EarlyStop, time_decorator
from lib.utils import mask_loss, compute_loss, predict_and_evaluate, contrast_loss

class AdapGLE2ETrainer(TFTrainer):
    def __init__(self, adj_filename, grid_node_map, risk_mask, data_type, model_pred, model_graph, optimizer_pred, optimizer_graph, scheduler_pred,
                 scheduler_graph, epoch_num, num_iter, scaler, model_save_path, lam, patience):
        super().__init__(model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler, model_save_path, patience)
        self.model_graph: nn.Module = model_graph
        self.optimizer_graph = optimizer_graph
        self.scheduler_graph = scheduler_graph
        self.num_iter: int = num_iter
        self.model_save_path: str = model_save_path
        self.grid_node_map = grid_node_map
        self.risk_mask = risk_mask
        self.lam = lam
        self.patience = patience
        self.data_type = data_type
        self.scaler =scaler
        self.device = next(self.model_graph.parameters()).device

        adj_filename = ','.join(adj_filename)
        adj_mx_list, self.risk_sim = self.__get_adj_mx_list(adj_filename.split(','))
        self.epsilon = 1 / adj_mx_list[0].size(0) * 0.5

        self.cur_adj_mx = self.update_best_adj_mx(adj_mx_list)

        self.batches_seen = 0

        self._model_save_dir = os.path.dirname(self.model_save_path)
        self._graph_save_path = os.path.join(self._model_save_dir, 'best_adj_mx.npy')
        self.early_stop = EarlyStop(patience, min_is_best=True)
        self._delta = 0.25


    def train_one_epoch(self, data_loader):
        self.model.train()
        self.model_graph.train()

        for train_feature, target_time, gragh_feature, train_label in data_loader:
            train_feature, target_time, gragh_feature, train_label = train_feature.to(self.device), target_time.to(self.device), gragh_feature.to(self.device), train_label.to(self.device)
            adj_mx = self.model_graph(self.cur_adj_mx)
            pred, encode = self.model(train_feature, target_time, gragh_feature, adj_mx, self.grid_node_map)

            loss = self.model_loss_func(pred, train_label, encode, target_time, adj_mx)
            self.optimizer.zero_grad()
            self.optimizer_graph.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_graph.step()
            self.batches_seen += 1

        return self.model_graph(self.cur_adj_mx).detach()

    @time_decorator
    def train(self, train_data_loader, eval_data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        tmp_state_save_path = os.path.join(self._model_save_dir, 'temp.pkl')
        min_loss = torch.finfo(torch.float32).max

        for epoch in range(1, self.max_epoch_num + 1):
            # train one epoch
            adj_mx = self.train_one_epoch(train_data_loader)

            # evaluate
            print('Epoch {}'.format(epoch), end='  ')
            eval_loss, _, _ = self.evaluate(eval_data_loader, metrics, adj_mx=adj_mx)

            # Criteria for early stopping
            if self.early_stop.reach_stop_criteria(eval_loss):
                break

            # save model state when meeting minimum loss
            # save to a temporary path first to avoid overwriting original state.
            if eval_loss < min_loss:
                torch.save(self.model.state_dict(), tmp_state_save_path)
                np.save(self._graph_save_path, adj_mx.cpu().numpy())
                min_loss = eval_loss

            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.scheduler_graph is not None:
                self.scheduler_graph.step()

        self.early_stop.reset()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)

    def test(self, data_loader, high_test_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        self.model.load_state_dict(torch.load(self.model_save_path))
        adj_mx = torch.tensor(
            data=np.load(self._graph_save_path),
            dtype=torch.float32,
            device=self.device
        )
        print('All time intervals Test results')
        self.evaluate(data_loader, adj_mx, metrics)
        print('High frequency traffic accident intervals Test results')
        self.evaluate(high_test_loader, adj_mx, metrics)

    def model_loss_func(self, y_pred, y_true, encode, target_time, cur_adj_mx=None):
        pred_loss = mask_loss(y_pred, y_true, self.risk_mask, self.data_type)
        if self.model.training:
            mx_p, mx_q = self.cur_adj_mx, cur_adj_mx
            mx_delta = torch.sign(mx_q) - torch.sign(mx_p)
            sim_loss = F.relu(F.relu(mx_delta).mean() - self._delta) / self._delta
            pred_loss = pred_loss + sim_loss + self.lam * contrast_loss(encode, self.risk_sim, target_time, self.grid_node_map)

        return pred_loss

    def __get_adj_mx_list(self, adj_path_list):
        adj_mx_list = []
        risk_sim = []
        for adj_path in adj_path_list:
            if 'risk' in adj_path:
                risk_sim.append(pkl.load(open(adj_path, 'rb')).astype(np.float32))
            adj_mx = get_adj(pkl.load(open(adj_path, 'rb')).astype(np.float32), 'gcn')
            adj_mx = torch.tensor(adj_mx, dtype=torch.float32, device=self.device)
            adj_mx_list.append(adj_mx)
        return adj_mx_list, risk_sim[0]


    def update_best_adj_mx(self, adj_mx_list):

        print('initialize adj')

        adj_mx_sum = torch.zeros_like(adj_mx_list[0])
        adj_num_sum = torch.zeros_like(adj_mx_sum)
        for adj_mx in adj_mx_list:
            adj_mx_sum += adj_mx
            adj_num_sum += (1 + torch.sign(adj_mx - 1e-4)) / 2  # （邻接矩阵的值大于等于0）0或者很小的数变为0；其余1
        adj_mx_sum /= adj_num_sum
        adj_mx_sum[torch.logical_or(torch.isnan(adj_mx_sum), torch.isinf(adj_mx_sum))] = 0
        best_adj_mx = torch.from_numpy(get_adj(adj_mx_sum.cpu().numpy().astype(np.float32), 'gcn')).to(self.device)

        return best_adj_mx
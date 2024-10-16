import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from adj_mx import get_adj
from .base import Trainer, TFTrainer
import pickle as pkl
from lib.utils import mask_loss, compute_loss, predict_and_evaluate, contrast_loss

class AdapGLTrainer(Trainer):
    """
    Adaptive Graph Learning Networks Trainer.

    Args:
        adj_mx_path: Paths of all adjacent matrixes which are splited by ','.
        model_pred: Model for prediction.
        model_graph: Model for graph learning.
        optimizer_pred: Optimizer for prediction model training.
        optimizer_graph: Optimizer for graph learning model training.
        scheduler_pred: Learning rate scheduler for prdiction model training.
        scheduler_graph: Learning rate scheduler for graph learning model training.
        epoch_num: Training epoch for prediction model and graph learning model per iteration.
        num_iter: Number of iteration.
        max_adj_num: The maximal volume of adj_mx set.
        scaler: Scaler for data set.
        model_save_path: Path to save of prediction model.
    """
    def __init__(self, adj_filename, grid_node_map, risk_mask, data_type, model_pred, model_graph, optimizer_pred, optimizer_graph, scheduler_pred,
                 scheduler_graph, epoch_num, num_iter, scaler, model_save_path, lam, patience):
        super().__init__(model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler, model_save_path)
        self.model_pred: nn.Module = model_pred#定义一个名为 self.model_pred 的属性，类型为 nn.Module，并将其初始化为 model_pred
        self.model_graph: nn.Module = model_graph
        self.num_iter: int = num_iter
        self.model_save_path: str = model_save_path
        self.grid_node_map = grid_node_map
        self.risk_mask = risk_mask
        self.lam = lam
        self.patience = patience
        self.data_type = data_type
        self.scaler =scaler
        self.device = next(self.model_pred.parameters()).device

        adj_filename = ','.join(adj_filename)
        self.adj_mx_list, self.risk_sim = self.__get_adj_mx_list(adj_filename.split(','))
        self.epsilon = 1 / self.adj_mx_list[0].size(0) * 0.5

        self.best_adj_mx = None
        self.update_best_adj_mx()

        model_save_dir, model_name = os.path.split(self.model_save_path)
        self.graph_save_path = os.path.join(model_save_dir, 'GRAPH.pkl')

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        self.model_pred_trainer = self.ModelPredTrainer(
            model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler,
            model_save_path, self)

        self.model_graph_trainer = self.GraphLearnTrainer(
            model_graph, optimizer_graph, scheduler_graph, epoch_num, scaler,
            self.graph_save_path, self)

        # my own variables.
        best_save_dir = os.path.join(model_save_dir, model_name.split('.')[0])
        self.best_pred_path = os.path.join(best_save_dir, model_name)
        self.best_graph_path = os.path.join(best_save_dir, 'best_adj_mx.npy')

        if not os.path.exists(best_save_dir):
            os.mkdir(best_save_dir)

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



    def update_best_adj_mx(self, new_adj=None):
        """
        Update self.best_adj_mx.

        criteria:
            - replace: use the newest subgraph as best_adj_mx;
            - union: combine the adj_mx in self.adj_mx_list;
            - weight_union: weighted sum of adj_mx in self.adj_mx_list according to
                evaluate loss.
        """

        if self.best_adj_mx == None and new_adj == None:
            print('initial adj')
            adj_mx_sum = torch.zeros_like(self.adj_mx_list[0])
            adj_num_sum = torch.zeros_like(adj_mx_sum)
            for adj_mx in self.adj_mx_list:
                adj_mx_sum += adj_mx
                adj_num_sum += (1 + torch.sign(adj_mx - 1e-4)) / 2  # （邻接矩阵的值大于等于0）0或者很小的数变为0；其余1
            adj_mx_sum /= adj_num_sum
            adj_mx_sum[torch.logical_or(torch.isnan(adj_mx_sum), torch.isinf(adj_mx_sum))] = 0
            best_adj_mx = torch.from_numpy(get_adj(adj_mx_sum.cpu().numpy().astype(np.float32), 'gcn')).to(self.device)
        else:
            best_adj_mx = new_adj
            print('got new adj')
            #print(new_adj[:5,:5])
        self.best_adj_mx = best_adj_mx.float()


    def update_num_epoch(self, cur_iter):
        if cur_iter == self.num_iter // 2 + 1 :
            self.model_pred_trainer.max_epoch_num += 5#总迭代次数 self.num_iter 的一半加一（self.num_iter // 2 + 1），则会将模型预测训练器（self.model_pred_trainer）的最大周期数（max_epoch_num）增加 5。

    def train_one_iteration(self, train_data_loader, eval_data_loader, metrics, cur_iter):
        self.model_pred_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_pred.load_state_dict(torch.load(self.model_save_path))
        self.model_graph_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_graph.load_state_dict(torch.load(self.graph_save_path))
        # Add the learned graph to self.adj_mx_list, update self.best_adj_mx.
        new_adj_mx = self.model_graph(self.best_adj_mx).detach()#取最好的新矩阵
        self.update_best_adj_mx(new_adj_mx)



    @torch.no_grad()
    def evaluate(self, data_loader, adj_mx):
        """
        Test the prediction loss on data set 'data_loader' using 'adj_mx'.
        """
        loss, _, _ = self.model_pred_trainer.evaluate(data_loader, adj_mx=adj_mx)
        return loss

    @torch.no_grad()
    def test(self, test_loader, high_test_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        self.model_pred.load_state_dict(torch.load(self.best_pred_path))
        best_adj_mx_np = np.load(self.best_graph_path)
        best_adj_mx = torch.tensor(
            data=best_adj_mx_np,
            dtype=torch.float32,
            device=self.device
        )
        sparsity = (best_adj_mx_np == 0).sum() / (best_adj_mx_np.shape[0] ** 2)
        print('Sparsity: {:.4f}'.format(sparsity))
        print('Test results of current graph: ')
        print('All time intervals Test results')
        _, y_true, y_pred = self.model_pred_trainer.evaluate(test_loader, best_adj_mx, metrics)
        self.model_pred_trainer.print_test_result(y_pred, y_true, self.scaler, self.risk_mask)
        print('High frequency traffic accident intervals Test results')
        _, y_true_high, y_pred_high = self.model_pred_trainer.evaluate(high_test_loader, best_adj_mx, metrics)
        self.model_pred_trainer.print_test_result(y_pred_high, y_true_high, self.scaler, self.risk_mask)

    def train(self, train_data_loader, eval_data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
        print('Start Training...')
        min_loss = torch.finfo(torch.float32).max
        wait = 0
        for i in range(self.num_iter):
            print('Iteration {}:'.format(i + 1))
            self.train_one_iteration(train_data_loader, eval_data_loader, metrics, i)
            print('Evaluation results of current graph:')
            cur_loss = self.evaluate(eval_data_loader, self.best_adj_mx)
            if cur_loss < min_loss:
                print(' save best model state because of meeting minimum loss')
                copyfile(self.model_save_path, self.best_pred_path)
                np.save(self.best_graph_path, self.best_adj_mx.cpu().numpy())
                min_loss = cur_loss
                print(f'Iteration {i+1}, got new min_loss {min_loss}')
                wait = 0
            else:
                wait += 1
                if wait == 3:
                    print('连续3次Iteration没有改进,Stop Training...')
                    break
            #self.update_num_epoch(i + 1)

    class ModelPredTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj.patience)
            self.outer_obj: AdapGLTrainer = outer_obj #self.outer_obj 是一个 AdapGLTrainer 类的实例，可以通过该实例来访问和操作 AdapGLTrainer 类中定义的属性和方法
            self.risk_mask = self.outer_obj.risk_mask
            self.risk_sim = self.outer_obj.risk_sim
            self.grid_node_map = self.outer_obj.grid_node_map
            self.lam = self.outer_obj.lam
            self.data_type = self.outer_obj.data_type
            self.num_iter: int = 1
            self.batches_seen: int = 0#用于跟踪已经处理过的批次数量

        def train_one_epoch(self, data_loader):
            self.model.train()
            adj_mx = self.outer_obj.best_adj_mx
            for _ in range(self.num_iter):
                for train_feature, target_time, gragh_feature, train_label in data_loader:
                    train_feature, target_time, gragh_feature, train_label = train_feature.to(self.device), target_time.to(
                        self.device), gragh_feature.to(self.device), train_label.to(self.device)

                    pred, encode = self.model(train_feature, target_time, gragh_feature, adj_mx, self.grid_node_map)

                    loss = self.model_loss_func(pred, train_label, self.risk_mask, self.data_type) + self.lam * self.contrastive_loss(encode, self.risk_sim, target_time, self.grid_node_map)
                    self.optimizer.zero_grad()
                    loss.backward()#计算网络中各个参数对损失的梯度
                    self.optimizer.step()
                    self.batches_seen += 1

        def train(self, train_data_loader, eval_data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            print('Round for prediction model:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, adj_mx = None, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            if adj_mx is None:
                adj_mx = self.outer_obj.best_adj_mx
            return super().evaluate(data_loader, adj_mx, metrics)

        @torch.no_grad()
        def test(self, data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true)

        @staticmethod
        def model_loss_func(y_pred, y_true, risk_mask, data_type):
            return mask_loss(y_pred, y_true, risk_mask, data_type)#L1 损失（也称为平均绝对误差）
        @staticmethod
        def contrastive_loss(encode, risk_sim, target_time, grid_node_map):
            return contrast_loss(encode, risk_sim, target_time, grid_node_map)


    class GraphLearnTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path,outer_obj.patience)
            self.outer_obj: AdapGLTrainer = outer_obj
            self.risk_mask = self.outer_obj.risk_mask
            self.grid_node_map = self.outer_obj.grid_node_map
            self.data_type = self.outer_obj.data_type
            self.num_iter: int = 1
            self.delta: int = 0.05
            self.best_adj_mx = self.outer_obj.best_adj_mx

        def train_one_epoch(self, data_loader):
            self.model.train()
            self.outer_obj.model_pred.train()
            #冻结 self.outer_obj.model_pred 的参数
            for param in self.outer_obj.model_pred.parameters():
                param.requires_grad = False
            for _ in range(self.num_iter):
                for feature, target_time, gragh_feature, label in data_loader:
                    feature, target_time, gragh_feature, label = feature.to(
                        self.device), target_time.to(
                        self.device), gragh_feature.to(self.device), label.to(self.device)
                    adj_mx = self.model(self.best_adj_mx)
                    #with torch.no_grad():
                    pred, _ = self.outer_obj.model_pred(feature, target_time, gragh_feature, adj_mx, self.grid_node_map)#用的是新生成的矩阵预测而不是当前最好矩阵
                    loss = self.model_loss_func(pred, label)#预测损失+保证生成矩阵稀疏性的损失
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # 检查和打印模型参数及其梯度
                    '''
                    print("Model parameter gradients:")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")
                        else:
                            print(f"Parameter: {name}, Gradient: None")
                    
                    print("Model parameter gradients222:")
                    for name, param in self.outer_obj.model_pred.named_parameters():
                        if param.grad is not None:
                            print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")
                        else:
                            print(f"Parameter: {name}, Gradient: None")
                    '''
            # 解除参数冻结状态，以备将来可能的需求
            for param in self.outer_obj.model_pred.parameters():
                param.requires_grad = True

        def train(self, train_data_loader, eval_data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            print('Round for graph learning:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            adj_mx = self.model(self.outer_obj.best_adj_mx).detach()
            return self.outer_obj.model_pred_trainer.evaluate(data_loader, adj_mx, metrics)#只用了新矩阵

        @torch.no_grad()
        def test(self, data_loader, metrics=('mask_rmse_np', 'Recall', 'MAP')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true)

        def model_loss_func(self, y_pred, y_true):
            """Loss function of Graph Learn Model."""
            mx_p = self.outer_obj.best_adj_mx
            mx_q = self.model(mx_p).to(self.device)#与之前算的其实一样adj_mx
            mx_delta = torch.sign(mx_q) - torch.sign(mx_p)
            sim_loss = F.relu(F.relu(mx_delta).mean() - self.delta) / self.delta
            pred_loss = mask_loss(y_pred, y_true, self.risk_mask, self.data_type)
            return pred_loss + sim_loss
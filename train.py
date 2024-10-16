import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import trainer
import numpy as np
import json
from time import time
from datetime import datetime
import argparse
import random

import sys
import os

'''
这里我们将纽约的数据作为案例进行注释，以便于读者的理解和可复线
'''
torch.autograd.set_detect_anomaly(True)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_adjacent, get_grid_node_map_maxtrix
from model.LG_STSSLNet import LG_STSSLNet, GSEM
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/chicago/Chicago_Config.json", help='configuration file')

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'device:{device}')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']    # (8760, 48, 20, 20)
mask_filename = config['mask_filename']            # (20, 20)
risk_mask = get_mask(mask_filename)

adj_filename = []
adj_filename.append(config['road_adj_filename'])    # (243, 243)
adj_filename.append(config['risk_adj_filename'])    # (243, 243)
if 'nyc' in config_filename:
    adj_filename.append(config['poi_adj_filename'])

grid_node_filename = config['grid_node_filename']  # (400, 243)
grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)  # 网格到有效具有路网节点的映射
num_of_vertices = grid_node_map.shape[1]  # 实际包含的路网有效区域

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

'''
基础参数，包括数据的划分比率、数据特征尺度、epoch等信息
'''
train_rate = config['train_rate']
valid_rate = config['valid_rate']
recent_prior = config['recent_prior']
day_prior = config['day_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior + day_prior  # 一条数据样本的时间步长



def main(config):
    batch_size = config['batch_size']
    pre_len = config['pre_len']
    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    grid_in_channel = config['grid_in_channel']
    num_conv_layer = config['num_conv_layer']
    num_gcn_layer = config['num_gcn_layer']
    num_head = config['num_head']
    num_of_graph_feature = config['num_of_graph_feature']
    num_of_graph_filters = config['num_of_graph_filters']
    num_of_conv_filters = config['num_of_conv_filters']
    emb_size = config['emb_size']
    num_time_features = config['num_time_features']
    cheb_k = config['cheb_k']
    emb_dim_final = config['emb_dim_final']
    K = config['K']
    Conv_K = config['Conv_K']
    grap_embe_dim = config['grap_embe_dim']

    loaders = []
    train_data_shape = ""
    graph_feature_shape = ""
    for idx, (x, y, target_times, high_x, high_y, high_target_times, scaler) in enumerate(
            normal_and_generate_dataset_time(
                    all_data_filename,
                    train_rate=train_rate,
                    valid_rate=valid_rate,
                    recent_prior=recent_prior,
                    day_prior=day_prior,
                    week_prior=week_prior,
                    one_day_period=one_day_period,
                    days_of_week=days_of_week,
                    pre_len=pre_len)):

        if 'nyc' in all_data_filename:
            graph_x = x[:, :, [0, 46, 47], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 46, 47], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)
        if 'chicago' in all_data_filename:
            graph_x = x[:, :, [0, 39, 40], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 39, 40], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)

        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
              "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))
        '''
        feature: (4584, 7, 48, 20, 20) label: (4584, 1, 20, 20) time: (4584, 32) high feature: (1337, 7, 48, 20, 20) high label: (1337, 1, 20, 20)
        graph_x: (4584, 7, 3, 243) high_graph_x: (1337, 7, 3, 243)
        '''
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape
        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(target_times),
                torch.from_numpy(graph_x),
                torch.from_numpy(y)
            ),
            batch_size=batch_size,
            shuffle=(idx == 0)
        ))
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_graph_x),
                    torch.from_numpy(high_y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            )
    train_loader, val_loader, test_loader = loaders


    DGSTNET_Model = LG_STSSLNet(grid_in_channel, num_of_vertices, num_time_features, num_conv_layer, num_of_conv_filters, num_gcn_layer, num_head, gru_hidden_size, num_of_gru_layers,
                 num_of_graph_feature, num_of_graph_filters,
                 seq_len, cheb_k, K, Conv_K, emb_size, emb_dim_final,
                 north_south_map, west_east_map, batch_size, pre_len)
    GraphLearn_Model = GSEM(num_of_vertices, grap_embe_dim)
    # multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        DGSTNET_Model = nn.DataParallel(DGSTNET_Model)
    DGSTNET_Model.to(device)
    GraphLearn_Model.to(device)
    print(DGSTNET_Model)

    num_of_parameters = 0
    for name, parameters in DGSTNET_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)


    Optimizer = getattr(sys.modules['torch.optim'], config['optimizer'])
    optimizer_pred = Optimizer(
        DGSTNET_Model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    optimizer_graph = Optimizer(GraphLearn_Model.parameters(), lr=config['learning_rate'])

    lr_scheduler = config['lr_scheduler']
    Scheduler = getattr(sys.modules['torch.optim.lr_scheduler'],
                        lr_scheduler)
    scheduler_pred = Scheduler(optimizer_pred, T_max=config['T_max'], eta_min=config['eta_min'])
    scheduler_graph = None

    # --------------------------- Train -------------------------
    #net_trainer = trainer.AdapGLTrainer(
    #    adj_filename, grid_node_map, risk_mask, config['data_type'], DGSTNET_Model, GraphLearn_Model, optimizer_pred, optimizer_graph,
    #    scheduler_pred, scheduler_graph, config['num_epoch'], config['num_iter'], scaler, config['model_save_path'], config['lam'], patience
    #)
    net_trainer = trainer.E2ETrainer(
         adj_filename, grid_node_map, risk_mask, config['data_type'], DGSTNET_Model, GraphLearn_Model, optimizer_pred, optimizer_graph,
             scheduler_pred, scheduler_graph, config['num_epoch'], config['num_iter'], scaler, config['model_save_path'], config['lam'], patience
     )

    net_trainer.train(train_loader, val_loader)
    net_trainer.test(test_loader, high_test_loader)


if __name__ == "__main__":
    #
    main(config)
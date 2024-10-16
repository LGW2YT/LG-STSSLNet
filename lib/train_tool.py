import time
import functools
import numpy as np
import torch


class EarlyStop:
    def __init__(self, tol_num, min_is_best):
        self.tol_num = tol_num
        self.min_is_best = min_is_best

        self.count, self.cur_values = None, None
        self.reset()

    def reset(self):
        self.count = 0
        finfo = np.finfo(np.float32)
        self.cur_best_value = finfo.max if self.min_is_best else finfo.min #如果 self.min_is_best 评估为 True，则将 self.cur_value 设置为 np.float32 的最大可表示值（从 finfo.max 获取）。否则，将其设置为最小可表示值（从 finfo.min 获取）

    def reach_stop_criteria(self, cur_value):
        if isinstance(cur_value, torch.Tensor):
            cur_value = cur_value.numpy()  # 将 PyTorch tensor 转换为 NumPy 数组
        elif isinstance(cur_value, float):
            cur_value = np.float32(cur_value)  # 将 float 转换为 NumPy float32
        if self.min_is_best:
            if cur_value >= self.cur_best_value:
                self.count = self.count + 1
            else:
                self.cur_best_value = cur_value
                self.count = 0
        else:
            if cur_value <= self.cur_best_value:
                self.count = self.count + 1
            else:
                self.cur_best_value = cur_value
                self.count = 0
        if self.count == self.tol_num:
            print('Early stop reached!')
            return True

        return False

def time_decorator(func):
    """ A decorator that shows the time consumption of method "func" """
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Time Consumption: {:.2f}s'.format(end_time - start_time))
        return result
    return inner
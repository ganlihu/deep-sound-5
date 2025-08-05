import os
import logging
import pickle
from glob import glob
from datetime import datetime as dt
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import sed_eval
import dcase_util

from data.utils import windows2events
from experiments import settings
from experiments.utils import set_random_init


logger = logging.getLogger('yaer')


class Experiment:
    ''' Base class to represent an experiment using audio signals (removed movement signals). '''
    def __init__(self,
                 model_factory,
                 features_factory,
                 X,
                 y,
                 window_width,
                 window_overlap,
                 name,
                 audio_sampling_frequency=6000,  # 适配新数据集的音频采样率
                 no_event_class='no-event',
                 manage_sequences=False,
                 model_parameters_grid={},
                 use_raw_data=False,
                 quantization=None,
                 data_augmentation=False):
        self.timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        self.model_factory = model_factory
        self.features_factory = features_factory
        self.X = X
        self.y = y
        self.window_width = window_width
        self.window_overlap = window_overlap
        self.name = name
        self.audio_sampling_frequency = audio_sampling_frequency
        self.no_event_class = no_event_class
        self.manage_sequences = manage_sequences
        self.model_parameters_grid = model_parameters_grid
        self.use_raw_data = use_raw_data
        self.train_validation_segments = []
        self.quantization = quantization
        self.data_augmentation = data_augmentation

        # 创建实验路径
        self.path = os.path.join(settings.experiments_path, name, self.timestamp)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # 配置日志
        logger.handlers = []
        fileHandler = logging.FileHandler(f"{self.path}/experiment.log")
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        # 设置随机种子
        set_random_init()

    def run(self):
        ''' 运行实验并保存相关信息 '''
        # 切换为新数据集 jaw_movements2020
        self.X = self.X['jaw_movements2020']
        self.y = self.y['jaw_movements2020']

        # 保留原有的5折交叉验证划分（假设新数据集的分段编号兼容）
        folds = {
            '1': [45, 3, 23, 2, 17],
            '2': [20, 42, 21, 1, 39],
            '3': [28, 22, 33, 51, 55],
            '4': [10, 40, 14, 41, 19],
            '5': [47, 24, 7, 18]
        }

        for i in folds.values():
            self.train_validation_segments.extend(i)

        hash_method_instance = hashlib.new('sha256')
        params_results = {}
        full_grid = list(ParameterGrid(self.model_parameters_grid))

        if len(full_grid) > 1:
            for params_combination in full_grid:
                if params_combination != {}:
                    logger.info('运行参数组合: %s 的交叉验证', params_combination)
                else:
                    logger.info('无参数网格，直接运行交叉验证')

                # 计算参数哈希用于结果比较
                hash_method_instance.update(str(params_combination).encode())
                params_combination_hash = hash_method_instance.hexdigest()

                params_combination_result = self.execute_kfoldcv(
                    folds=folds,
                    is_grid_search=True,
                    parameters_combination=params_combination)

                # 保存参数组合及结果
                params_results[params_combination_hash] = (params_combination_result,
                                                           params_combination)

            best_params_combination = max(params_results.values(), key=lambda i: i[0])[1]
            logger.info('-' * 25)
            logger.info('所有参数组合结果: %s', str(params_results))
            logger.info('-' * 25)
            logger.info('最佳参数组合: %s', best_params_combination)
        else:
            logger.info('-' * 25)
            logger.info('无参数网格，跳过网格搜索')
            best_params_combination = full_grid[0]

        self.execute_kfoldcv(
            folds=folds,
            is_grid_search=False,
            parameters_combination=best_params_combination)

    def execute_kfoldcv(self,
                        folds,
                        is_grid_search,
                        parameters_combination):
        ''' 执行k折交叉验证 '''
        signal_predictions = {}

        for ix_fold, fold in folds.items():
            logger.info('运行第 %s 折交叉验证', ix_fold)

            # 划分训练集和测试集
            test_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) in fold]
            train_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) not in fold]
            train_fold_keys = [k for k in train_fold_keys if \
                int(k.split('_')[1]) in self.train_validation_segments]

            logger.info('训练集片段: %s', str(train_fold_keys))
            X_train = []
            y_train = []
            for train_signal_key in train_fold_keys:
                if self.manage_sequences:
                    X_train.append(self.X[train_signal_key])
                    y_train.append(self.y[train_signal_key])
                else:
                    X_train.extend(self.X[train_signal_key])
                    y_train.extend(self.y[train_signal_key])

            if self.data_augmentation:
                from augly.audio import functional
                # 计算类别分布
                all_y = []
                n_labels = 0
                for i_file in range(len(X_train)):
                    for i_window in range(len(X_train[i_file])):
                        if y_train[i_file][i_window] != 'no-event':
                            all_y.append(y_train[i_file][i_window])
                            n_labels += 1
                unique, counts = np.unique(all_y, return_counts=True)
                classes_probs = dict(zip(unique, counts / n_labels))

                # 复制训练样本用于数据增强
                import copy
                X_augmented = copy.deepcopy(X_train)
                y_augmented = copy.deepcopy(y_train)

                # 仅处理音频通道（移除IMU通道处理）
                for i_file in range(len(X_train)):
                    during_event = False
                    discard_event = False
                    for i_window in range(len(X_train[i_file])):
                        window_label = y_train[i_file][i_window]
                        
                        if window_label == 'no-event':
                            during_event = False
                            discard_event = False
                        elif not during_event and window_label not in ['no-event',
                                                                       'bite',
                                                                       'rumination-chew']:
                            during_event = True
                            # 对多数类事件进行随机丢弃
                            if np.random.rand()
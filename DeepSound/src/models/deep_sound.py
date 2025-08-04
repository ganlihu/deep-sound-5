import os
import copy
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping


# 初始化模块日志
logger = logging.getLogger('yaer')


class DeepSoundBaseRNN:
    ''' Create a RNN with robust data handling '''
    def __init__(self,
                 batch_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 validation_split=0.2,
                 results_dir="training_results",
                 models_dir="trained_models"):
        self.classes_ = None
        self.padding_class = None
        self.max_seq_length = None  # 动态确定的最大序列长度
        self.feature_dim = None     # 特征维度

        # 核心参数
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.ghost_dim = 2
        self.padding = "valid"
        self.training_shape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.validation_split = validation_split

        # 路径配置
        self.results_dir = results_dir
        self.models_dir = models_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _preprocess_data(self, data, is_label=False):
        """统一处理特征或标签数据，转换为数组并调整维度"""
        processed = []
        for item in data:
            try:
                if isinstance(item, list):
                    # 处理列表类型（支持嵌套列表）
                    arr = np.array(item, dtype='float32' if not is_label else 'int')
                elif isinstance(item, np.ndarray):
                    # 处理数组类型
                    arr = item.astype('float32' if not is_label else 'int')
                else:
                    raise ValueError(f"不支持的数据类型: {type(item)}")

                # 调整标签维度（确保1D）
                if is_label and arr.ndim > 1:
                    arr = arr.flatten()
                # 调整特征维度（确保2D）
                if not is_label and arr.ndim == 1:
                    arr = arr.reshape(-1, 1)

                processed.append(arr)
            except Exception as e:
                logger.warning(f"数据预处理失败: {str(e)}，使用默认值")
                default_len = 100 if self.max_seq_length is None else self.max_seq_length
                default_feat = 1 if self.feature_dim is None else self.feature_dim
                default_val = self.padding_class if is_label else -100.0
                processed.append(np.full((default_len, default_feat) if not is_label else (default_len,), 
                                        default_val, dtype='float32' if not is_label else 'int'))
        return processed

    def _pad_data(self, data, target_length, pad_value, is_label=False):
        """统一填充数据至目标长度"""
        padded = []
        for arr in data:
            current_len = arr.shape[0]
            if current_len < target_length:
                # 计算填充宽度
                pad_width = (0, target_length - current_len) if is_label else ((0, target_length - current_len), (0, 0))
                arr_padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=pad_value)
            else:
                arr_padded = arr[:target_length] if is_label else arr[:target_length, :]
            padded.append(arr_padded)
        return padded

    def fit(self, X, y):
        ''' Train network based on given data with robust handling '''
        # 统一数据格式为列表
        if not isinstance(X, list):
            X = [X]
        if not isinstance(y, list):
            y = [y]

        # 预处理特征和标签
        processed_X = self._preprocess_data(X, is_label=False)
        processed_y = self._preprocess_data(y, is_label=True)

        # 确定最大序列长度
        self.max_seq_length = max(max(arr.shape[0] for arr in processed_X), 
                                 max(arr.shape[0] for arr in processed_y))
        self.feature_dim = processed_X[0].shape[1] if processed_X else 1
        logger.info(f"自动确定序列长度: {self.max_seq_length}, 特征维度: {self.feature_dim}")

        # 填充特征和标签至统一长度
        X_pad = self._pad_data(processed_X, self.max_seq_length, -100.0, is_label=False)
        self.classes_ = list(set(np.concatenate([arr.flatten() for arr in processed_y])))
        self.padding_class = len(self.classes_) if self.classes_ else 0
        y_pad = self._pad_data(processed_y, self.max_seq_length, self.padding_class, is_label=True)

        # 转换为模型输入格式
        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y_pad).astype('float32')
        logger.info(f"训练数据形状 - X: {X.shape}, y: {y.shape}")

        # 调整验证集比例（小样本保护）
        num_samples = X.shape[0]
        actual_val_split = self.validation_split
        if num_samples < 5:
            logger.warning(f"样本量较少 ({num_samples}个)，调整验证集比例")
            actual_val_split = 0.0 if num_samples == 1 else max(1/num_samples, min(0.1, self.validation_split))

        # 配置回调函数
        callbacks = [
            EarlyStopping(patience=50, monitor='val_loss' if actual_val_split > 0 else 'loss'),
            CSVLogger(os.path.join(self.results_dir, "training_metrics.csv"), append=False)
        ]

        # 样本权重计算
        sample_weights = self._get_samples_weights(y) if self.set_sample_weights else None

        # 模型训练
        history = self.model.fit(
            x=X,
            y=y,
            epochs=self.n_epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=actual_val_split,
            shuffle=True,
            sample_weight=sample_weights,
            callbacks=callbacks
        )

        # 保存最终模型
        self.model.save(os.path.join(self.models_dir, "final_model.h5"))
        logger.info(f"模型已保存至 {self.models_dir}")
        return history

    def predict(self, X):
        """预测函数（优化输入一致性）"""
        # 预处理输入
        processed_X = self._preprocess_data(X, is_label=False)
        X_pad = self._pad_data(processed_X, self.max_seq_length, -100.0, is_label=False)
        X = np.asarray(X_pad).astype('float32')

        # 特征缩放（保持原始逻辑）
        if self.feature_scaling and X.size > 0:
            X = (X + 1.0) * 100

        # 模型预测
        y_pred = self.model.predict(X).argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        """预测概率函数"""
        processed_X = self._preprocess_data(X, is_label=False)
        X_pad = self._pad_data(processed_X, self.max_seq_length, -100.0, is_label=False)
        X = np.asarray(X_pad).astype('float32')

        if self.feature_scaling and X.size > 0:
            X = (X + 1.0) * 100

        return self.model.predict(X)

    def _get_samples_weights(self, y):
        """改进的样本权重计算（覆盖所有类别）"""
        all_labels = np.ravel(y).astype(int)
        # 确保包含所有可能类别（包括填充类）
        all_classes = np.unique(np.concatenate([all_labels, [self.padding_class]]))
        class_counts = np.bincount(all_labels, minlength=len(all_classes))
        # 处理零计数类别
        class_counts = np.where(class_counts == 0, 1, class_counts)
        total = len(all_labels)
        weights = total / (len(all_classes) * class_counts)
        # 填充类权重设为0
        weights[all_classes == self.padding_class] = 0.0
        # 映射到每个样本
        sample_weights = np.array([weights[int(label)] for label in all_labels])
        return sample_weights.reshape(y.shape)

    def clear_params(self):
        """重置模型参数（用于交叉验证）"""
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepSound(DeepSoundBaseRNN):
    def __init__(self,
                 batch_size=5,
                 input_size=4000,
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 conv_layers_config=None,
                 gru_units=128,
                 dense_units=[256, 128],
                 **kwargs):
        ''' Create network instance of DeepSound architecture with configurable layers '''
        super().__init__(batch_size=batch_size,
                         n_epochs=n_epochs,
                         training_reshape=training_reshape,
                         set_sample_weights=set_sample_weights,
                         feature_scaling=feature_scaling,** kwargs)

        # 卷积层配置（默认使用原始结构，支持自定义）
        self.conv_layers_config = conv_layers_config or [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.input_size = input_size
        self.output_size = output_size

        # 构建模型
        self.model = self._build_model()
        self.weights_ = copy.deepcopy(self.model.get_weights())

    def _build_model(self):
        """构建可配置的DeepSound模型（保留原始架构核心）"""
        # 卷积子网络
        cnn = Sequential(name="cnn_backbone")
        cnn.add(layers.Rescaling(1./255, input_shape=(None, self.input_size)))  # 输入标准化

        for ix_l, layer in enumerate(self.conv_layers_config):
            filters, kernel, stride, activation = layer
            # 每个卷积块包含2个卷积层
            for _ in range(2):
                cnn.add(layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel,
                    strides=stride,
                    activation=activation,
                    padding=self.padding,
                    data_format=self.data_format
                ))
            # 除最后一个块外添加Dropout
            if ix_l < (len(self.conv_layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2))

        # 卷积后处理
        cnn.add(layers.MaxPooling1D(4))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dropout(rate=0.2))

        # 全连接子网络
        ffn = Sequential(name="ffn_head")
        for units in self.dense_units:
            ffn.add(layers.Dense(units, activation=activations.relu))
            ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(self.output_size, activation=activations.softmax))

        # 整体模型（时序+双向GRU）
        model = Sequential([
            layers.InputLayer(input_shape=(None, self.input_size, 1), name='input_layer'),
            layers.TimeDistributed(cnn),
            layers.Bidirectional(layers.GRU(
                self.gru_units,
                activation="tanh",
                return_sequences=True,
                dropout=0.2
            ), name="bidirectional_gru"),
            layers.TimeDistributed(ffn)
        ])

        # 编译模型
        model.compile(
            optimizer=Adagrad(),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )
        return model
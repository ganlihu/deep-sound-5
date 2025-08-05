import os
import copy

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


class AblationBase:
    def _preprocess(self, X, y=None, training=False):
        X_pad = []

        # 仅处理音频数据（原X中的第一个元素）
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        if training:
            y = keras.preprocessing.sequence.pad_sequences(
                y,
                padding='post',
                value=self.padding_class,
                dtype=object)

        # 只保留音频数据（移除所有IMU相关的X_acc和X_gyr）
        X_sound = [X_pad[0]]

        if training:
            sequences_length = 46
            X_sound = np.asarray(X_sound).astype('float32')
            X_sound = np.moveaxis(X_sound, 0, -1)

            if self.training_reshape:
                # 仅调整音频数据形状
                new_shape = (int(X_sound.shape[0] * X_sound.shape[1] // sequences_length),
                             sequences_length,
                             X_sound.shape[2],
                             X_sound.shape[3])
                X_sound = np.resize(X_sound, new_shape)
                y = np.resize(y, new_shape[:2])

            y = np.asarray(y).astype('float32')
        else:
            X_sound = np.asarray(X_sound).astype('float32')
            X_sound = np.moveaxis(X_sound, 0, -1)

        # 仅返回音频数据和标签（移除IMU相关返回值）
        return X_sound, y

    def predict_proba(self, X):
        return self.model.predict(X,
                                  batch_size=self.batch_size)

    def _get_samples_weights(self, y):
        # 获取类别计数（移除IMU相关逻辑）
        unique_values, counts = np.unique(np.ravel(y),
                                          return_counts=True)

        class_weight = {value: count
                        for value, count
                        in zip(unique_values, counts)
                        if value != self.padding_class}

        # 根据样本数量设置权重（样本少的类别权重高）
        max_count = np.max(list(class_weight.values()))
        class_weight = {k: max_count / v for k, v in class_weight.items()}

        # 填充类别的权重设为0
        class_weight[self.padding_class] = 0

        # 为每个样本分配权重
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in class_weight.items():
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepFusionAblationA(AblationBase):
    ''' 仅基于声音信号的消融模型（移除所有IMU相关层） '''
    def __init__(self,
                 batch_size=5,
                 input_size_audio=(None, 1800, 1),  # 仅保留音频输入尺寸
                 output_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' 创建仅处理音频的模型实例 '''
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.conv_layers_padding = "valid"
        self.training_reshape = training_reshape
        self.set_sample_weights = set_sample_weights
        dropout_rate = 0.0

        # 仅定义音频输入
        deep_sound_input = Input(shape=input_size_audio)

        # 声音处理CNN
        sound_cnn = Sequential()
        sound_cnn.add(layers.Rescaling())
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=18,
                                    strides=3,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(32,
                                    kernel_size=9,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Dropout(rate=dropout_rate))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.Conv1D(128,
                                    kernel_size=3,
                                    strides=1,
                                    activation=activations.relu,
                                    padding=self.conv_layers_padding,
                                    data_format=self.data_format))
        sound_cnn.add(layers.MaxPooling1D(4))
        sound_cnn.add(layers.Flatten())

        sound_x = layers.TimeDistributed(sound_cnn)(deep_sound_input)

        # RNN层处理时序特征
        rnn = layers.Bidirectional(layers.GRU(256,
                                              activation=activations.relu,
                                              return_sequences=True,
                                              dropout=dropout_rate))(sound_x)

        # 全连接层分类
        dense = Sequential()
        dense.add(Dense(256, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(128, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))
        dense.add(Dense(64, activation='relu'))
        dense.add(layers.Dropout(rate=dropout_rate))

        dense_td_model = layers.TimeDistributed(dense)(rnn)
        output = Dense(output_size, activation=activations.softmax)(dense_td_model)

        # 仅使用音频输入构建模型
        model = Model(inputs=deep_sound_input, outputs=output)
        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())
        # 添加模型保存路径（确保与训练逻辑兼容）
        self.output_path_model_checkpoints = os.path.join('models', 'checkpoints')
        self.output_logs_path = os.path.join('logs')
        os.makedirs(self.output_path_model_checkpoints, exist_ok=True)
        os.makedirs(self.output_logs_path, exist_ok=True)

    def fit(self, X, y):
        ''' 仅使用音频数据训练模型 '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)

        # 仅获取音频数据和标签
        X_sound, y = self._preprocess(X, y, training=True)

        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=150),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path_model_checkpoints, 'model.tf'),
                save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=self.output_logs_path)
        ]

        # 样本权重（仅基于音频标签计算）
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        self.model.fit(x=X_sound,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        ''' 仅使用音频数据进行预测 '''
        X_sound, _ = self._preprocess(X)
        y_pred = self.model.predict(X_sound).argmax(axis=-1)
        return y_pred
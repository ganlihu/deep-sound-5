# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd
from scipy import signal
import librosa
import more_itertools

from data.cache_manager import DatasetCache
from data import utils_data_sources as utils


logger = logging.getLogger(__name__)


def get_windows_from_audio_signal(audio_signal, sampling_frequency, window_width, window_overlap):
    """从音频信号中提取窗口"""
    window_size = int(sampling_frequency * window_width)
    step_size = int(window_size * (1 - window_overlap))
    windows = []
    for i in range(0, len(audio_signal) - window_size + 1, step_size):
        windows.append(audio_signal[i:i+window_size])
    return windows


def get_window_labels(df_labels, window_starts, window_ends, threshold, no_event_class):
    """为窗口分配标签"""
    labels = []
    for s, e in zip(window_starts, window_ends):
        overlap = df_labels.apply(
            lambda row: max(0, min(e, row['end']) - max(s, row['start'])), axis=1
        )
        valid = overlap / (e - s) >= threshold
        if valid.any():
            labels.append(df_labels[valid]['jm_event'].iloc[0])
        else:
            labels.append(no_event_class)
    return labels


def main(data_source_names=['jaw_movements2020'],  # 适配缓存中的数据集名称
         audio_sampling_frequency=6000,  # 与缓存采样率一致
         window_width=0.3,  # 与缓存窗口宽度一致
         window_overlap=0.5,
         label_overlapping_threshold=0.5,
         filter_noises=True,
         no_event_class_name='no-event',
         filters=None,
         invalidate_cache=False):
    """
    处理音频数据生成窗口化数据集（移除所有IMU相关逻辑）
    """
    logger = logging.getLogger(__name__)

    cache = DatasetCache()

    # 缓存键移除所有IMU相关参数
    cache_item = cache.load(
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        no_event_class_name=no_event_class_name,
        filters=filters
    )

    if cache_item and not invalidate_cache:
        logger.info('*** 从缓存加载数据集 ***')
        (X, y) = cache_item
        return X, y

    logger.info('*** 从头创建数据集 ***')
    available_datasets = utils.list_datasets()

    # 验证数据集是否存在
    for data_source_name in data_source_names:
        assert data_source_name in available_datasets, \
            f'数据集 {data_source_name} 不存在'

    # 仅保留音频相关的断言检查
    assert (audio_sampling_frequency * window_width) % 5 == 0, \
        '音频采样率和窗口宽度不兼容（需满足 (采样率 * 窗口宽度) % 5 == 0）'

    assert (audio_sampling_frequency * window_width * (1 - window_overlap)) % 5 == 0, \
        '音频采样率和窗口重叠率不兼容（需满足 (采样率 * 窗口宽度 * (1-重叠率)) % 5 == 0）'

    X = {}
    y = {}

    for dataset in data_source_names:
        segment_files = utils.get_files_in_dataset(available_datasets[dataset])

        X_dataset_segments = {}
        y_dataset_segments = {}
        for segment in segment_files:
            # segment结构：[音频文件, 标签文件]（已移除IMU文件）
            segment_name = os.path.basename(segment[0]).split('.')[0]
            logger.info(f"> 处理片段: {segment_name}")

            # 读取并重采样音频
            audio_signal, sf = librosa.load(segment[0])
            audio_signal = librosa.resample(
                y=audio_signal,
                orig_sr=sf,
                target_sr=audio_sampling_frequency
            )

            # 应用滤波器（仅处理音频）
            if filters:
                for filter in filters:
                    filter_method, channels, _ = filter  # 忽略IMU标志
                    audio_signal = filter_method(audio_signal)

            # 读取并处理标签
            df_segment_labels = pd.read_csv(
                segment[-1],  # 标签文件在segment的最后一位
                sep='\t',
                names=["start", "end", "jm_event"]
            )

            # 标签映射（保持原逻辑）
            general_mask = df_segment_labels.jm_event
            df_segment_labels.loc[general_mask == 'u', 'jm_event'] = 'unknown'
            df_segment_labels.loc[general_mask == 'b', 'jm_event'] = 'bite'
            df_segment_labels.loc[general_mask == 'c', 'jm_event'] = 'grazing-chew'
            df_segment_labels.loc[general_mask == 'r', 'jm_event'] = 'rumination-chew'
            df_segment_labels.loc[general_mask == 'x', 'jm_event'] = 'chewbite'

            # 提取音频窗口
            audio_windows = get_windows_from_audio_signal(
                audio_signal,
                sampling_frequency=audio_sampling_frequency,
                window_width=window_width,
                window_overlap=window_overlap
            )

            # 计算窗口时间戳并生成标签
            window_step = window_width * (1 - window_overlap)
            window_starts = np.arange(0, len(audio_windows) * window_step, window_step)
            window_ends = window_starts + window_width

            window_labels = get_window_labels(
                df_segment_labels,
                window_starts,
                window_ends,
                label_overlapping_threshold,
                no_event_class_name
            )

            # 存储窗口数据（仅保留音频）
            X_dataset_segments[segment_name] = [[window] for window in audio_windows]
            y_dataset_segments[segment_name] = window_labels

        X[dataset] = X_dataset_segments
        y[dataset] = y_dataset_segments

    # 保存到缓存（仅包含音频相关参数）
    cache.save(
        X, y,
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        no_event_class_name=no_event_class_name,
        filters=filters
    )

    return X, y
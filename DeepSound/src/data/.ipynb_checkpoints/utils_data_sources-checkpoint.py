from collections import namedtuple
import os
import glob

from data.settings import DATA_SOURCES_PATH


def list_datasets():
    """Return a dictionary with available datasets (仅包含音频相关配置)."""

    # 仅保留音频相关字段，移除所有IMU相关配置
    Dataset = namedtuple("dataset",
                         ["str_id", "name", "folder", "audio_files_format",
                          "audio_sampling_frequency"])

    assert os.path.exists(DATA_SOURCES_PATH), f"Path {DATA_SOURCES_PATH} does not exist."

    datasets = {
        # 修正原zavalla2022为jm2020，统一数据集命名
        'jm2020': Dataset(
            str_id="jm2020",
            name="Jaw Movement 2020 Dataset (仅音频数据)",
            folder=os.path.join(DATA_SOURCES_PATH, "jm2020"),  # 匹配实际文件夹路径
            audio_files_format="wav",
            audio_sampling_frequency=6000  # 与缓存中采样率一致
        )
    }

    return datasets


def get_files_in_dataset(dataset):
    """获取数据集中的文件列表（仅包含音频和标签文件）."""

    if dataset.audio_files_format == "wav":
        ext = "wav"

    dataset_files = []
    # 查找所有标签文件（匹配labels文件夹下的txt文件）
    labels_file_list = sorted(glob.glob(os.path.join(dataset.folder, "labels", "*.txt")))

    for label_file in labels_file_list:
        # 从标签文件名提取录音ID（如recording_01.txt -> recording_01）
        recording_id = os.path.splitext(os.path.basename(label_file))[0]
        # 构建对应的音频文件路径（audios文件夹下的wav文件）
        audio_file = os.path.join(dataset.folder, "audios", f"{recording_id}.{ext}")
        
        # 文件组仅包含音频和标签文件
        files_group = [audio_file, label_file]

        # 检查文件是否存在
        for file in files_group:
            assert os.path.isfile(file), f'未找到文件: {file}'

        dataset_files.append(tuple(files_group))

    return dataset_files
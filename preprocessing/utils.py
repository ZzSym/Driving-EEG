import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from omegaconf import DictConfig
from scipy.signal import decimate

def load_npz_files(npz_dir: str) -> tuple[list, list, list]:
    """
    Load the data from the .npz files in the output directory.

    Args:
    ----
        output_dir (str): The directory containing the .npz files.

    Returns:
    -------
        data_list (list): A list of data arrays from the .npz files.
        times_list (list): A list of time arrays from the .npz files.
        sfreq_list (list): A list of sampling frequencies from the .npz files.
    """

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

    data_list = []
    times_list = []
    sfreq_list = []

    for file in npz_files:
        file_path = os.path.join(npz_dir, file)
        npzfile = np.load(file_path)
        data_list.append(npzfile['data'])
        times_list.append(npzfile['times'])
        sfreq_list.append(npzfile['sfreq'])
    
    return data_list, times_list, sfreq_list

def load_log_file(file_path: str) -> dict:
    """
    Load log file.

    Args:
    ----
        file_path (str): File path.

    Returns:
    -------
        log_dict: Dict of log data, including timepoint, throttle, brake, steering,
            angular_velocity, location, velocity, and acceleration.

    """
    log = pd.read_csv(file_path).values
    start_time = log[0][0]
    timepoint = []
    throttle = []
    brake = []
    steering = []
    angular_velocity = []
    location = []
    velocity = []
    acceleration = []
    for row in log:
        timepoint.append(get_ms(start_time, row[0]))
        throttle.append(row[1])
        brake.append(row[2])
        steering.append(row[3])
        angular_velocity.append(row[4:7])
        location.append(row[7:10])
        velocity.append(row[10:13])
        acceleration.append(row[13:16])
    log_dict = {
        "timepoint": np.array(timepoint),
        "throttle": np.array(throttle),
        "brake": np.array(brake),
        "steering": np.array(steering),
        "angular_velocity": np.array(angular_velocity),
        "location": np.array(location),
        "velocity": np.array(velocity),
        "acceleration": np.array(acceleration),
    }
    return log_dict

def get_ch_names(file_path: str) -> list:
    """
    Get channel names.

    Args:
    ----
        file_path (str): File path.

    Returns:
    -------
        list (List): List of channel names.

    """
    ch_names = []
    raw = mne.io.read_raw_fif(file_path, preload=True)
    ch_names = raw.ch_names
    return ch_names

def get_ms(time_0: str, time_1: str) -> int:
    """
    Get time difference in milliseconds.

    Args:
    ----
        time_0 (str): Time 0. Time str format: "%Y-%m-%d-%H-%M-%S-%f".
        time_1 (str): Time 1. Time str format: "%Y-%m-%d-%H-%M-%S-%f".

    Returns:
    -------
        int: Time difference in milliseconds.

    """
    datetime_0 = datetime.strptime(time_0, "%Y-%m-%d-%H-%M-%S-%f")
    datetime_1 = datetime.strptime(time_1, "%Y-%m-%d-%H-%M-%S-%f")
    return int((datetime_1 - datetime_0).total_seconds() * 1000)

def align_data(log_dict: dict, data: np.ndarray, cfg: DictConfig) -> dict:
    """
    Align and preprocess eeg data.

    Args:
    ----
        log_dict (dict): Dict of log data.
        data (np.ndarray): EEG data.
        cfg (DictConfig): Config.

    Returns:
    -------
        dict: Dict of aligned data including timepoint, features, steering, location, velocity.

    """
    aligned_data = {"timepoint": [], "feature": [], "steering": [], "location": [], "velocity": []}
    for i in tqdm(range(len(log_dict["timepoint"]))):
        time_point = log_dict["timepoint"][i]
        start = time_point - cfg.time_window + cfg.time_bias - cfg.padding_before
        end = time_point + cfg.time_bias + cfg.padding_after
        if start >= 0 and end <= len(data[0]):
            decimate_sequence = decimate(data[:, start:end], cfg.decimate_rate, ftype=cfg.filter_type, axis=1)
            aligned_data["timepoint"].append(time_point)
            aligned_data["feature"].append(
                decimate_sequence[
                    :,
                    int(cfg.padding_before / cfg.decimate_rate) : -int(cfg.padding_after / cfg.decimate_rate),
                ]
            )
            aligned_data["steering"].append(log_dict["steering"][i])
            aligned_data["location"].append(log_dict["location"][i])
            aligned_data["velocity"].append(log_dict["velocity"][i])
    aligned_data["timepoint"] = np.array(aligned_data["timepoint"])
    aligned_data["feature"] = np.array(aligned_data["feature"])
    aligned_data["steering"] = np.array(aligned_data["steering"])
    aligned_data["location"] = np.array(aligned_data["location"])
    aligned_data["velocity"] = np.array(aligned_data["velocity"])
    return aligned_data

def align_preprocessed_data(log_dict: dict, data: np.ndarray, cfg: DictConfig) -> dict:
    """
    Align preprocessed eeg data.

    Args:
    ----
        log_dict (dict): Dict of log data.
        data (np.ndarray): EEG data.
        cfg (DictConfig): Config.

    Returns:
    -------
        dict: Dict of aligned data including timepoint, features, steering, location, velocity.

    """
    aligned_data = {"timepoint": [], "feature": [], "steering": [], "location": [], "velocity": []}
    for i in tqdm(range(len(log_dict["timepoint"]))):
        time_point = log_dict["timepoint"][i]
        start = int((time_point - cfg.time_window + cfg.time_bias - cfg.padding_before) / cfg.decimate_rate)
        end = int((time_point + cfg.time_bias + cfg.padding_after) / cfg.decimate_rate)
        if start >= 0 and end <= len(data[0]):
            decimate_sequence = data[:, start:end]
            aligned_data["timepoint"].append(time_point)
            aligned_data["feature"].append(
                decimate_sequence[
                    :,
                    int(cfg.padding_before / cfg.decimate_rate) : -int(cfg.padding_after / cfg.decimate_rate),
                ]
            )
            aligned_data["steering"].append(log_dict["steering"][i])
            aligned_data["location"].append(log_dict["location"][i])
            aligned_data["velocity"].append(log_dict["velocity"][i])
    aligned_data["timepoint"] = np.array(aligned_data["timepoint"])
    aligned_data["feature"] = np.array(aligned_data["feature"])
    aligned_data["steering"] = np.array(aligned_data["steering"])
    aligned_data["location"] = np.array(aligned_data["location"])
    aligned_data["velocity"] = np.array(aligned_data["velocity"])
    return aligned_data

def sample_wise_grouping(
    cfg: DictConfig, aligned_datas: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group patches by the time interval specified in the config.

    Args:
    ----
        cfg (DictConfig): Config object.
        aligned_datas (List[Dict[str, np.ndarray]]): Aligned data dictionary.

    Returns:
    -------
        feature (np.ndarray, shape (n_patches, n_channel, n_time)): Features of each patch.
        timepoint (np.ndarray, shape (n_patches,)): Timepoints of each patch.
        session_mask (np.ndarray, shape (n_patches,)): Session index of each patch.
        groups (np.ndarray, shape (n_patches,)): Group index of each patch.
        steering (np.ndarray, shape (n_patches,)): Steering of each patch.
        location (np.ndarray, shape (n_patches, 3)): Location of each patch.
        velocity (np.ndarray, shape (n_patches, 3)): Velocity of each patch.
    """
    group_interval = cfg.model_fitting.group_interval
    feature = []
    timepoint = []
    session_mask = []
    groups = []
    steering = []
    location = []
    velocity = []
    group_index = 0
    for i in range(len(aligned_datas)):
        aligned_data = aligned_datas[i]
        feature.append(np.array([feature for feature in aligned_data["feature"]]))
        timepoint.append(aligned_data["timepoint"])
        steering.append(aligned_data["steering"])
        location.append(aligned_data["location"])
        velocity.append(aligned_data["velocity"])
        session_mask.append(np.ones(len(aligned_data["steering"])) * i)
        last_timepoint = 0
        for tp in aligned_data["timepoint"]:
            if tp - last_timepoint > group_interval:
                group_index += 1
                last_timepoint = tp
            groups.append(group_index)
        group_index += 1

    feature = np.concatenate(feature)
    timepoint = np.concatenate(timepoint)
    session_mask = np.concatenate(session_mask)
    groups = np.array(groups)
    steering = np.concatenate(steering)
    location = np.concatenate(location)
    velocity = np.concatenate(velocity)
    return feature, timepoint, session_mask, groups, steering, location, velocity

def z_score_norm(feature: np.ndarray) -> np.ndarray:
    """
    Z-score normalization at time dimension.

    Args:
    ----
        feature (np.ndarray): Feature matrix with shape (n_patches, n_channels, n_time).

    Returns:
    --------
        norm_feature (np.ndarray): Normalized feature matrix with shape (n_patches, n_channels, n_time).
    """
    mean_time_dim = np.mean(feature, axis=2, keepdims=True)
    std_time_dim = np.std(feature, axis=2, keepdims=True)
    norm_feature = (feature - mean_time_dim) / std_time_dim
    return norm_feature
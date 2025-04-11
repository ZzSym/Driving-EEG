import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_npz_files_default() -> list:
    """
    Load the chosen npz files in the given directory and return a list of dictionaries

    Returns:
    -------
        list (List[Dict[str, np.ndarray]]): List of dictionaries containing the data from the npz files
    """
    aligned_datas = [
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_1.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_2.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_3.npz", allow_pickle=True),
        # np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_4.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_5.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_6.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_7.npz", allow_pickle=True),
        np.load("../data/aligned_data/aligned_data_20250224/20240819_jmd_aligned_8.npz", allow_pickle=True),
    ]
    for i in range(len(aligned_datas)):
        aligned_datas[i] = {key: value for key, value in aligned_datas[i].items()}
    return aligned_datas

def load_npz_files(directory_path: str) -> list:
    """
    Load all the npz files in the given directory and return a list of dictionaries

    Args:
    ----
        directory_path (str): Path to the directory containing npz files

    Returns:
    -------
        list (List[Dict[str, np.ndarray]]): List of dictionaries containing the data from the npz files
    """
    npz_files = [f for f in os.listdir(directory_path) if f.endswith('.npz')]
    data_list = []
    for file in npz_files:
        file_path = os.path.join(directory_path, file)
        with np.load(file_path, allow_pickle=True) as npz_file:
            data_list.append({key: npz_file[key] for key in npz_file})
    return data_list

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

def session_wise_grouping(
    aligned_datas: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group patches by session.

    Args:
    ----
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
    feature = []
    timepoint = []
    session_mask = []
    groups = []
    steering = []
    location = []
    velocity = []
    for i in range(len(aligned_datas)):
        aligned_data = aligned_datas[i]
        feature.append(np.array([feature for feature in aligned_data["feature"]]))
        timepoint.append(aligned_data["timepoint"])        
        steering.append(aligned_data["steering"])
        location.append(aligned_data["location"])
        velocity.append(aligned_data["velocity"])
        session_mask.append(np.ones(len(aligned_data["steering"])) * i)

    feature = np.concatenate(feature)
    timepoint = np.concatenate(timepoint)
    session_mask = np.concatenate(session_mask)
    groups = np.array(session_mask)
    steering = np.concatenate(steering)
    location = np.concatenate(location)
    velocity = np.concatenate(velocity)
    return feature, timepoint, session_mask, groups, steering, location, velocity

def scenario_wise_grouping(
    cfg: DictConfig, aligned_datas: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group patches by the scenario of time interval specified in the config.

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
        group_index = 0

    feature = np.concatenate(feature)
    timepoint = np.concatenate(timepoint)
    session_mask = np.concatenate(session_mask)
    groups = np.array(groups)
    steering = np.concatenate(steering)
    location = np.concatenate(location)
    velocity = np.concatenate(velocity)
    return feature, timepoint, session_mask, groups, steering, location, velocity

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

def make_binary_label(input: np.ndarray) -> np.ndarray:
    """"
    Make binary label from input(steering).

    Args:
    ----
        input (np.ndarray): Steering data.

    Returns:
    -------
        binary_label (np.ndarray): Binary label. If steering > 0, binary_label = 1, else binary_label = 0.
    """
    binary_label = np.zeros(input.shape)
    binary_label[input > 0] = 1
    return binary_label

def make_multi_label(input: np.ndarray, split_list: list) -> np.ndarray:
    """
    Make multi label from input(steering).

    Args:
    ----
        input (np.ndarray): Steering data.
        split_list (list): List of split values.

    Returns:
    -------
        label (np.ndarray): Multi label splited by split_list.
    """
    label = np.zeros(input.shape[0])
    for i in range(len(split_list) - 1):
        label[(input >= split_list[i]) & (input < split_list[i + 1])] = i
    return label

def load_data_model(npz_dir: str) -> list[dict[str, np.ndarray]]:
    """
    Load data for modeling from npz files.

    Returns:
    -------
        data_list (List[Dict[str, np.ndarray]]): List of dictionaries containing the train and test data from the npz files
    """
    data_list = np.load(npz_dir, allow_pickle=True)
    data_list = {key: value for key, value in data_list.items()}
    return data_list

def evaluate_prediction(
    prediction: list[np.ndarray], labels: np.ndarray, train_mask: list[np.ndarray], test_mask: list[np.ndarray]
):
    """
    Evaluate prediction.

    Args:
    ----
        prediction (List[np.ndarray]): List of predictions of each validation split.
        labels (np.ndarray): Labels.
        train_mask (List[np.ndarray]): Train mask of each validation split.
        test_mask (List[np.ndarray]): Test mask of each validation split.

    """
    train_r2_score = []
    train_rmse = []
    train_ae = []
    test_r2_score = []
    test_rmse = []
    test_ae = []
    for model_index in range(len(train_mask)):
        train_r2_score.append(
            r2_score(np.array(labels)[train_mask[model_index]], prediction[model_index][train_mask[model_index]])
        )
        train_rmse.append(
            np.sqrt(
                mean_squared_error(
                    np.array(labels)[train_mask[model_index]], prediction[model_index][train_mask[model_index]]
                )
            )
        )
        train_ae.append(
            mean_absolute_error(
                np.array(labels)[train_mask[model_index]], prediction[model_index][train_mask[model_index]]
            )
        )
        test_r2_score.append(
            r2_score(np.array(labels)[test_mask[model_index]], prediction[model_index][test_mask[model_index]])
        )
        test_rmse.append(
            np.sqrt(
                mean_squared_error(
                    np.array(labels)[test_mask[model_index]], prediction[model_index][test_mask[model_index]]
                )
            )
        )
        test_ae.append(
            mean_absolute_error(
                np.array(labels)[test_mask[model_index]], prediction[model_index][test_mask[model_index]]
            )
        )
    print(f"Train R2 Score: {np.mean(train_r2_score):.4f} +- {np.std(train_r2_score):.4f}")
    print(f"Train RMSE: {np.mean(train_rmse):.4f} +- {np.std(train_rmse):.4f}")
    print(f"Train AE: {np.mean(train_ae):.4f} +- {np.std(train_ae):.4f}")
    print(f"Test R2 Score: {np.mean(test_r2_score):.4f} +- {np.std(test_r2_score):.4f}")
    print(f"Test RMSE: {np.mean(test_rmse):.4f} +- {np.std(test_rmse):.4f}")
    print(f"Test AE: {np.mean(test_ae):.4f} +- {np.std(test_ae):.4f}")

def visualize_prediction(
    session_index: int, prediction: np.ndarray, labels: np.ndarray, session_mask: np.ndarray, train_mask: np.ndarray
):
    """
    Visualize the prediction and error distribution of a session.

    Args:
    ----
        session_index (int): The index of the session to visualize.
        prediction (np.ndarray): The prediction of the model.
        labels (np.ndarray): The ground truth labels.
        session_mask (np.ndarray): The session mask.
        train_mask (np.ndarray): The train mask.

    """
    session_bool = session_mask == session_index
    abs_residuals = np.abs(labels - prediction)
    condition = np.ones_like(abs_residuals, dtype=bool)
    condition[train_mask] = False
    change_points = np.where(np.diff(condition[session_bool]))[0] + 1
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(prediction[session_bool], label="predict")
    axs[0].plot(labels[session_bool], label="true")
    axs[0].legend()
    axs[0].set_title("Steering Angle Prediction")
    for change_point in change_points:
        axs[0].axvline(x=change_point, color="red", linestyle="--")
    axs[1].fill_between(
        np.arange(len(abs_residuals[session_bool])),
        abs_residuals[session_bool],
        where=condition[session_bool],
        color="red",
        alpha=0.3,
    )
    axs[1].fill_between(
        np.arange(len(abs_residuals[session_bool])),
        abs_residuals[session_bool],
        where=~condition[session_bool],
        color="green",
        alpha=0.3,
    )
    axs[1].set_title("Absolute Residuals")
    plt.show()
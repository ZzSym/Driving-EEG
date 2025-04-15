# @FileName: data_split.py
# @Author: ZzSym
# Divide the data in npz format into training datasets, validation sets, and test sets, and save them in data_model folder in npz format

import os 
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from omegaconf import OmegaConf
from utils import (
    sample_wise_grouping,
    z_score_norm,
)
import matplotlib.pyplot as plt

cfg = OmegaConf.create()
cfg.model_fitting = {}
cfg.model_fitting.group_interval = 16000

aligned_datas = [
    np.load("data/preprocessed/sub-jimingda/aligned_data_20250311/aligned_session1.npz", allow_pickle=True),
    np.load("data/preprocessed/sub-jimingda/aligned_data_20250311/aligned_session3.npz", allow_pickle=True),
    np.load("data/preprocessed/sub-jimingda/aligned_data_20250311/aligned_session5.npz", allow_pickle=True),
    np.load("data/preprocessed/sub-jimingda/aligned_data_20250311/aligned_session7.npz", allow_pickle=True),
]
for i in range(len(aligned_datas)):
    aligned_datas[i] = {key: value for key, value in aligned_datas[i].items()}

feature, timepoint, session_mask, groups, steering, location, velocity = sample_wise_grouping(cfg, aligned_datas)
feature = z_score_norm(feature)

print("feature shape: ", feature.shape)
print("session_mask: ", session_mask)
print("groups: ", groups)
print("groups_s1:",groups[session_mask==0])
print("groups_s2:",groups[session_mask==1])
print("groups_s3:",groups[session_mask==2])
print("groups_s4:",groups[session_mask==3])

# -----------------------------------------------------------------
# # make forward1.npz

# test_groups_index = [4, 14, 24, 34]
# test_indices = np.isin(groups, test_groups_index)
# train_indices = ~test_indices

# X_train = feature[train_indices]
# y_train = steering[train_indices]
# X_test = feature[test_indices]
# y_test = steering[test_indices]

# print(f"训练集特征形状: {X_train.shape}, 训练集标签形状: {y_train.shape}")
# print(f"测试集特征形状: {X_test.shape}, 测试集标签形状: {y_test.shape}")

# save_dir = "data/data_model/sub-jimingda"
# os.makedirs(save_dir, exist_ok=True)

# np.savez(os.path.join(save_dir, "forward1.npz"), 
#          feature_train=X_train,
#          steering_train=y_train,
#          feature_test=X_test,
#          steering_test=y_test,
#          session_mask=session_mask,
#          groups=groups,
#          location=location,
#          velocity=velocity,
#          timepoint=timepoint,
#          train_indices=train_indices,
#          test_indices=test_indices,
# )
# print(f"数据已保存到目录: {save_dir}")

# x = location[:, 0]
# y = location[:, 1] 
# z = location[:, 2]
# plt.figure(figsize=(10, 8))
# # plt.scatter(x, y, color='blue')
# plt.scatter(x[session_mask == 0], y[session_mask == 0], color='blue')
# # bend1_start = 2000
# # bend1_end = 2300
# # plt.scatter(x[bend1_start:bend1_end], y[bend1_start:bend1_end], color='red')
# # plt.scatter(x[test_indices], y[test_indices], color='yellow')
# plt.scatter(x[test_indices & (session_mask == 0)], y[test_indices & (session_mask == 0)], color='red')
# plt.title('Location')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# # plt.savefig('fig/session1_test.png', dpi=300)

# -----------------------------------------------------------------
# make forward2.npz

test_indices = np.isin(session_mask, 3)
train_indices = ~test_indices
X_train = feature[train_indices]
y_train = steering[train_indices]
X_test = feature[test_indices]
y_test = steering[test_indices]
print(f"训练集特征形状: {X_train.shape}, 训练集标签形状: {y_train.shape}")
print(f"测试集特征形状: {X_test.shape}, 测试集标签形状: {y_test.shape}")

save_dir = "data/data_model/sub-jimingda"
np.savez(os.path.join(save_dir, "forward2.npz"), 
         feature_train=X_train,
         steering_train=y_train,
         feature_test=X_test,
         steering_test=y_test,
         session_mask=session_mask,
         groups=groups,
         location=location,
         velocity=velocity,
         timepoint=timepoint,
         train_indices=train_indices,
         test_indices=test_indices,
)
print(f"数据已保存到目录: {save_dir}")

# -----------------------------------------------------------------

# 10-fold cross-validation
# n_splits = 10
# group_kfold = GroupKFold(n_splits=n_splits)
# train_mask = []
# test_mask = []
# for train_val_index, test_index in group_kfold.split(feature, steering, groups):
#     train_mask.append(train_val_index)
#     test_mask.append(val_index)

#     X_train, X_val = feature[train_index], feature[val_index]
#     y_train, y_val = label[train_index], label[val_index]


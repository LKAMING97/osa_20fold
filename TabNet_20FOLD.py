# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:TabNet_20FOLD.py
@Time:2022/7/25 15:17

"""
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.focal_loss import *
from core.config import *
import pandas as pd

if __name__ == "__main__":

    # 八个特征
    data = pd.read_csv("data/process_osa_v2.csv")
    features = data.iloc[:, 0:-1].astype(float)
    label = data.iloc[:, -1].astype(int)

    all_auc = list()
    all_lost = list()
    seed_everything(2022)
    kf = StratifiedKFold(n_splits=20, shuffle=True)

    for fold_n, (train_index, test_index) in enumerate(kf.split(features, label)):
        print("fold {}".format(fold_n + 1))
        features = np.array(features)

        X_train, y_train = features[train_index], np.array(label)[train_index].astype(int)
        X_test, y_test = features[test_index], np.array(label)[test_index].astype(int)

        # 训练验证集 按照75:20比例划分
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2105, stratify=y_train)

        batch_size = 128
        max_epochs = 150

        my_loss_fn = MultiFocalLoss(num_class=2)

        model = TabNetClassifier(optimizer_fn=torch.optim.AdamW,
                                 optimizer_params=dict(lr=4e-3),
                                 # optimizer_params=dict(lr=2e-2),
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                 scheduler_params={"step_size": 20,  # how to use learning rate scheduler
                                                   "gamma": 0.9},
                                 mask_type='sparsemax',  # 'entmax',  # 'entmax',
                                 n_steps=3,
                                 n_shared=3,
                                 n_independent=3,
                                 )

        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'val'],
            eval_metric=["accuracy", "auc"],
            max_epochs=max_epochs, patience=30,
            batch_size=batch_size,
            virtual_batch_size=16,
            num_workers=0,
            weights=1,
            drop_last=False,
            loss_fn=my_loss_fn
        )

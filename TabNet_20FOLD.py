# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:TabNet.py
@Time:2022/7/8 15:55

"""
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.focal_loss import *
from core.config import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":

    # 八个特征
    data = pd.read_csv("data/seven_feat_221018.csv")
    features = data.iloc[:, 0:-1].astype(float)
    label = data.iloc[:, -1].astype(int)

    all_auc = list()
    all_lost = list()
    seed_everything(2022)
    kf = StratifiedKFold(n_splits=20, shuffle=True)

    # auc
    all_train_auc = list()
    all_valid_auc = list()
    all_test_auc = list()
    # acc
    all_train_acc = list()
    all_valid_acc = list()
    all_test_acc = list()
    # sensitivity
    all_train_sensitivity = list()
    all_valid_sensitivity = list()
    all_test_sensitivity = list()
    # specificity
    all_train_specificity = list()
    all_valid_specificity = list()
    all_test_specificity = list()
    # ppv
    all_train_ppv = list()
    all_valid_ppv = list()
    all_test_ppv = list()
    # npv
    all_train_npv = list()
    all_valid_npv = list()
    all_test_npv = list()
    # 画平均ROC曲线的两个参数
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)

    for fold_n, (train_index, test_index) in enumerate(kf.split(features, label)):
        print("fold {}".format(fold_n + 1))
        features = np.array(features)
        threshold = 0.5
        train_result, val_result, test_result = [], [], []
        X_train, y_train = features[train_index], np.array(label)[train_index].astype(int)
        X_test, y_test = features[test_index], np.array(label)[test_index].astype(int)

        # 训练验证集 按照75:20比例划分
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2105, stratify=y_train)

        batch_size = 128
        max_epochs = 150

        my_loss_fn = MultiFocalLoss(num_class=2)

        model = TabNetClassifier(optimizer_fn=torch.optim.AdamW,
                                 optimizer_params=dict(lr=3e-3),
                                 # optimizer_params=dict(lr=2e-2),
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                 scheduler_params={"step_size": 20,  # how to use learning rate scheduler
                                                   "gamma": 0.8},
                                 mask_type="entmax",  # 'entmax',
                                 n_d=8,
                                 n_a=8,
                                 n_steps=3,
                                 n_independent=2,
                                 n_shared=2

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

        # 保存AUC最大的模型
        # save tabnet model
        # saving_path_name = "./tabnet_model_test"
        # if not os.path.exists(saving_path_name):
        #     os.makedirs(saving_path_name)
        # saved_filepath = model.save_model(saving_path_name)
        # plot losses
        plt.plot(model.history['loss'])
        plt.savefig("Tabnet_loss/" + "loss_" + str(fold_n) + ".jpg")
        plt.close()

        # train_predict

        tr_preds = model.predict_proba(X_train)
        train_auc = roc_auc_score(y_true=y_train, y_score=tr_preds[:, 1])
        for pred in tr_preds[:,1]:
            train_result.append(1) if pred > threshold else train_result.append(0)
        confusion = confusion_matrix(y_true=y_train, y_pred=train_result)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        # "ACC"
        train_acc = round((TP + TN) / (TP + FP + FN + TN), 3)
        # "Sensitivity: "
        train_Se = round(TP / (TP + FN + 0.01), 3)
        # "Specificity: "
        train_Sp = round(1 - (FP / (FP + TN + 0.01)), 3)
        # "Positive predictive value: "
        train_ppv = round(TP / (TP + FP + 0.01), 3)
        # "Negative predictive value: "
        train_npv = round(TN / (FN + TN + 0.01), 3)

        # val_predict
        val_preds = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_score=val_preds[:, 1], y_true=y_val)
        for pred in val_preds[:,1]:
            val_result.append(1) if pred > threshold else val_result.append(0)
        confusion = confusion_matrix(y_true=y_val, y_pred=val_result)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        # "ACC"
        val_acc = round((TP + TN) / (TP + FP + FN + TN), 3)
        # "Sensitivity: "
        val_Se = round(TP / (TP + FN + 0.01), 3)
        # "Specificity: "
        val_Sp = round(1 - (FP / (FP + TN + 0.01)), 3)
        # "Positive predictive value: "
        val_ppv = round(TP / (TP + FP + 0.01), 3)
        # "Negative predictive value: "
        val_npv = round(TN / (FN + TN + 0.01), 3)

        # test predict
        test_preds = model.predict_proba(X_test)
        test_auc = roc_auc_score(y_true=y_test, y_score=test_preds[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, test_preds[:, 1])

        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

        for pred in test_preds[:,1]:
            test_result.append(1) if pred > threshold else test_result.append(0)
        confusion = confusion_matrix(y_true=y_test, y_pred=test_result)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        # 预测训练集和验证集
        test_acc = round((TP + TN) / (TP + FP + FN + TN), 3)
        # "Sensitivity: "
        test_Se = round(TP / (TP + FN), 3)
        # "Specificity: "
        test_Sp = round(1 - (FP / (FP + TN)), 3)
        # "Positive predictive value: "
        test_ppv = round(TP / (TP + FP), 3)
        # "Negative predictive value: "
        test_npv = round(TN / (FN + TN), 3)

        # auc
        all_train_auc.append(train_auc)
        all_valid_auc.append(val_auc)

        # acc
        all_train_acc.append(train_acc)
        all_valid_acc.append(val_acc)

        # SP
        all_train_specificity.append(train_Sp)
        all_valid_specificity.append(val_Sp)

        # SE
        all_train_sensitivity.append(train_Se)
        all_valid_sensitivity.append(val_Se)

        # ppv
        all_train_ppv.append(train_ppv)
        all_valid_ppv.append(val_ppv)

        # npv
        all_train_npv.append(train_npv)
        all_valid_npv.append(val_npv)

        all_test_auc.append(test_auc)
        all_test_acc.append(test_acc)
        all_test_specificity.append(test_Sp)
        all_test_sensitivity.append(test_Se)
        all_test_ppv.append(test_ppv)
        all_test_npv.append(test_npv)
        print("-" * 40)

        # preds_valid = model.predict_proba(X_val)
        # valid_auc = roc_auc_score(y_true=y_val, y_score=preds_valid[:, 1])

        # print(f"FINAL VALID SCORE FOR  : {model.history['val_auc'][-1]}")
        print(f"FINAL TEST SCORE FOR   : {test_auc}")

        val_auc = max(model.history['val_auc'])

        all_auc.append(test_auc)

        print("*" * 40)

    mean_tpr /= kf.n_splits  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)
    assert round(mean_auc, 2) == round(np.mean(all_test_auc), 2)
    arr =list()
    arr.append(mean_tpr)
    arr.append(mean_fpr)
    arr.append(mean_auc)
    np.save("Tabnetmodel/tabnet.npy", arr)

    # 训练集
    # auc
    print("-" * 40)
    auc_mean = np.mean(all_train_auc)
    auc_std = np.std(all_train_auc)
    auc_conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_AUC:{all_train_auc}")
    print(f"MEAN_TRAIN_AUC:{np.mean(all_train_auc)}")
    print(f"train_AUC置信区间:{auc_conf_intveral}")

    # acc
    acc_mean = np.mean(all_train_acc)
    acc_std = np.std(all_train_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_ACC:{all_train_acc}")
    print(f"MEAN_TRAIN_ACC:{np.mean(all_train_acc)}")
    print(f"train_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_train_specificity)
    sp_std = np.std(all_train_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_SP:{all_train_specificity}")
    print(f"MEAN_TRAIN_SP:{np.mean(all_train_specificity)}")
    print(f"train_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_train_sensitivity)
    se_std = np.std(all_train_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_SE:{all_train_sensitivity}")
    print(f"MEAN_TRAIN_SE:{np.mean(all_train_sensitivity)}")
    print(f"train_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_train_ppv)
    ppv_std = np.std(all_train_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_ppv:{all_train_ppv}")
    print(f"MEAN_TRAIN_ppv:{np.mean(all_train_ppv)}")
    print(f"train_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_train_npv)
    npv_std = np.std(all_train_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std/np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_npv:{all_train_npv}")
    print(f"MEAN_TRAIN_npv:{np.mean(all_train_npv)}")
    print(f"train_npv置信区间:{npv_conf_intveral}")

    # 验证集指标
    print("-" * 40)
    auc_mean = np.mean(all_valid_auc)
    auc_std = np.std(all_valid_auc)
    auc_conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_AUC:{all_valid_auc}")
    print(f"MEAN_VALID_AUC:{np.mean(all_valid_auc)}")
    print(f"valid_AUC置信区间:{auc_conf_intveral}")

    # acc
    acc_mean = np.mean(all_valid_acc)
    acc_std = np.std(all_valid_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_ACC:{all_valid_acc}")
    print(f"MEAN_valid_ACC:{np.mean(all_valid_acc)}")
    print(f"valid_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_valid_specificity)
    sp_std = np.std(all_valid_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_SP:{all_valid_specificity}")
    print(f"MEAN_VALID_SP:{np.mean(all_valid_specificity)}")
    print(f"valid_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_valid_sensitivity)
    se_std = np.std(all_valid_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_SE:{all_valid_sensitivity}")
    print(f"MEAN_VALID_SE:{np.mean(all_valid_sensitivity)}")
    print(f"valid_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_valid_ppv)
    ppv_std = np.std(all_valid_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_ppv:{all_valid_ppv}")
    print(f"MEAN_VALID_ppv:{np.mean(all_valid_ppv)}")
    print(f"valid_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_valid_npv)
    npv_std = np.std(all_valid_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std/np.sqrt(kf.n_splits))
    print(f"ALL_VALID_npv:{all_valid_npv}")
    print(f"MEAN_VALID_npv:{np.mean(all_valid_npv)}")
    print(f"VALID_npv置信区间:{npv_conf_intveral}")

    # 测试集
    print("-" * 40)
    auc_mean = np.mean(all_test_auc)
    auc_std = np.std(all_test_auc)
    conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std/np.sqrt(kf.n_splits))
    print(f"ALL_AUC:{all_test_auc}")
    print(f"MEAN_TEST_AUC:{sum(all_test_auc) / len(all_test_auc)}")
    # print(f"STD_TEST_AUC:{np.std(all_test_auc)}")
    print(f"test_AUC置信区间:{conf_intveral}")

    # acc
    acc_mean = np.mean(all_test_acc)
    acc_std = np.std(all_test_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std/np.sqrt(kf.n_splits))
    print(f"ALL_TEST_ACC:{all_test_acc}")
    print(f"MEAN_TEST_ACC:{np.mean(all_test_acc)}")
    print(f"test_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_test_specificity)
    sp_std = np.std(all_test_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std/np.sqrt(kf.n_splits))
    print(f"ALL_test_SP:{all_test_specificity}")
    print(f"MEAN_TEST_SP:{np.mean(all_test_specificity)}")
    print(f"test_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_test_sensitivity)
    se_std = np.std(all_test_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std/np.sqrt(kf.n_splits))
    print(f"ALL_test_SE:{all_test_sensitivity}")
    print(f"MEAN_TEST_SE:{np.mean(all_test_sensitivity)}")
    print(f"test_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_test_ppv)
    ppv_std = np.std(all_test_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std/np.sqrt(kf.n_splits))
    print(f"ALL_test_ppv:{all_test_ppv}")
    print(f"MEAN_TEST_ppv:{np.mean(all_test_ppv)}")
    print(f"test_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_test_npv)
    npv_std = np.std(all_test_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std/np.sqrt(kf.n_splits))
    print(f"ALL_test_npv:{all_test_npv}")
    print(f"MEAN_test_npv:{np.mean(all_test_npv)}")
    print(f"test_npv置信区间:{npv_conf_intveral}")
    print("-" * 40)

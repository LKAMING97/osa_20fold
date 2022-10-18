# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:SVM_20FOLD.py
@Time:2022/7/21 14:14

"""
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.config import *
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


def grid_features(params, train_data, train_label, val_data, val_label):
    train_result, val_result = [], []
    best_valid_auc = 0.0
    values_range = list(params.values())
    # 进行多次网格搜索
    for C in values_range[0]:
        for gamma in values_range[1]:
            # 搭建模型
            model = SVC(kernel='rbf', gamma=gamma, class_weight='balanced', probability=True, C=C,
                        verbose=False)
            model.fit(train_data, train_label)
            preds_valid = model.predict_proba(val_data)
            valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=val_label)
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                # 保存模型
                joblib.dump(model, "model/svm_model/svm_bst_auc" + str(fold_n) + ".dat")
                print("最优AUC参数：C: %s ,  gamma: %s" % (
                    C, gamma))
    load_new_model = joblib.load("model/svm_model/svm_bst_auc" + str(fold_n) + ".dat")
    # 训练集指标
    preds_train = load_new_model.predict_proba(train_data)
    train_auc = roc_auc_score(y_score=preds_train[:, 1], y_true=train_label)
    threshold = 0.5
    for pred in preds_train[:,1]:
        train_result.append(1) if pred > threshold else train_result.append(0)
    # train_acc = accuracy_score(y_true=train_label, y_pred=preds_train)
    confusion = confusion_matrix(y_true=train_label, y_pred=train_result)
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
    # 验证集指标
    preds_val = load_new_model.predict_proba(val_data)
    val_auc = roc_auc_score(y_score=preds_val[:, 1], y_true=val_label)
    # val_acc = accuracy_score(y_true=val_label, y_pred=preds_val)
    threshold = 0.5
    for pred in preds_val[:,1]:
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

    return train_auc, train_acc, train_Sp, train_Se, train_ppv, train_npv, val_auc, val_acc, val_Sp, val_Se, val_ppv, val_npv


if __name__ == "__main__":
    data = pd.read_csv("data/test_data.csv")

    features = data.iloc[:, 0:-1].astype(float)
    label = data.iloc[:, -1].astype(int)
    seed_everything(2022)
    kf = StratifiedKFold(n_splits=20, shuffle=True)

    params = {'C': [1e-3, 1e-2, 1e-1, 1, 10],  # C越大,每个样本都要分对,易过拟合
              'gamma': [1e-4, 1e-3, 1e-2, 1e-1,1e-5]  # gamma越大,支持向量个数越少,高斯分布越高越瘦
              }

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
    # 针对不平衡数据集的权重设置
    weights = len(label[label == 0]) / len(label[label == 1])
    # 搜索参数，设置不同组参数进行网格搜索,选择最优
    all_auc = list()

    for fold_n, (train_index, test_index) in enumerate(kf.split(features, label)):
        test_result = []
        print("fold {}".format(fold_n + 1))
        features = np.array(features)

        X_train, y_train = features[train_index], np.array(label)[train_index].astype(int)
        X_test, y_test = features[test_index], np.array(label)[test_index].astype(int)

        # 训练集验证集75:20
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2105, stratify=y_train)

        # TODO:

        # 数据归一化
        scaler = StandardScaler()
        new_X_train = scaler.fit_transform(X_train)
        new_X_val = scaler.transform(X_val)
        new_X_test = scaler.transform(X_test)

        tr_auc, tr_acc, tr_Sp, tr_Se, tr_ppv, tr_npv, \
        valid_auc, valid_acc, valid_Sp, valid_Se, valid_ppv, valid_npv = grid_features(params, new_X_train, y_train, new_X_val, y_val)

        # auc
        all_train_auc.append(tr_auc)
        all_valid_auc.append(valid_auc)

        # acc
        all_train_acc.append(tr_acc)
        all_valid_acc.append(valid_acc)

        # SP
        all_train_specificity.append(tr_Sp)
        all_valid_specificity.append(valid_Sp)

        # SE
        all_train_sensitivity.append(tr_Se)
        all_valid_sensitivity.append(valid_Se)

        # ppv
        all_train_ppv.append(tr_ppv)
        all_valid_ppv.append(valid_ppv)

        # npv
        all_train_npv.append(tr_npv)
        all_valid_npv.append(valid_npv)

        # 读取该折最优模型进行预测
        load_model = joblib.load("model/svm_model/svm_bst_auc" + str(fold_n) + ".dat")
        preds_test = load_model.predict_proba(new_X_test)
        test_auc = roc_auc_score(y_score=preds_test[:, 1], y_true=y_test)
        fpr, tpr, thresholds = roc_curve(y_test, preds_test[:,1])

        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
        # test_auc = roc_auc_score(y_score=preds_test, y_true=y_test)
        # test_acc = accuracy_score(y_true=y_test, y_pred=preds_test)
        threshold = 0.5
        for pred in preds_test[:,1]:
            test_result.append(1) if pred > threshold else test_result.append(0)
        confusion = confusion_matrix(y_true=y_test, y_pred=test_result)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        test_acc = round((TP + TN) / (TP + FP + FN + TN), 3)
        # "Sensitivity: "
        test_Se = round(TP / (TP + FN + 0.01), 3)
        # "Specificity: "
        test_Sp = round(1 - (FP / (FP + TN + 0.01)), 3)
        # "Positive predictive value: "
        test_ppv = round(TP / (TP + FP + 0.01), 3)
        # "Negative predictive value: "
        test_npv = round(TN / (FN + TN + 0.01), 3)
        print(f"FINAL TEST SCORE FOR   : {test_auc}")
        all_test_auc.append(test_auc)
        all_test_acc.append(test_acc)
        all_test_specificity.append(test_Sp)
        all_test_sensitivity.append(test_Se)
        all_test_ppv.append(test_ppv)
        all_test_npv.append(test_npv)
        print("-" * 40)

    mean_tpr /= kf.n_splits  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    mean_auc = auc(mean_fpr, mean_tpr)
    assert round(mean_auc, 2) == round(np.mean(all_test_auc), 2)
    arr = list()
    arr.append(mean_tpr)
    arr.append(mean_fpr)
    arr.append(mean_auc)
    np.save("model/svm_model/svm.npy", arr)
    print("save npy done")

    # 计算置信区间
    print("-" * 40)
    auc_mean = np.mean(all_train_auc)
    auc_std = np.std(all_train_auc)
    auc_conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_AUC:{all_train_auc}")
    print(f"MEAN_TRAIN_AUC:{np.mean(all_train_auc)}")
    print(f"train_AUC置信区间:{auc_conf_intveral}")

    # acc
    acc_mean = np.mean(all_train_acc)
    acc_std = np.std(all_train_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_ACC:{all_train_acc}")
    print(f"MEAN_TRAIN_ACC:{np.mean(all_train_acc)}")
    print(f"train_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_train_specificity)
    sp_std = np.std(all_train_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_SP:{all_train_specificity}")
    print(f"MEAN_TRAIN_SP:{np.mean(all_train_specificity)}")
    print(f"train_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_train_sensitivity)
    se_std = np.std(all_train_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_SE:{all_train_sensitivity}")
    print(f"MEAN_TRAIN_SE:{np.mean(all_train_sensitivity)}")
    print(f"train_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_train_ppv)
    ppv_std = np.std(all_train_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_ppv:{all_train_ppv}")
    print(f"MEAN_TRAIN_ppv:{np.mean(all_train_ppv)}")
    print(f"train_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_train_npv)
    npv_std = np.std(all_train_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std / np.sqrt(kf.n_splits))
    print(f"ALL_TRAIN_npv:{all_train_npv}")
    print(f"MEAN_TRAIN_npv:{np.mean(all_train_npv)}")
    print(f"train_npv置信区间:{npv_conf_intveral}")

    # 验证集指标
    print("-" * 40)
    auc_mean = np.mean(all_valid_auc)
    auc_std = np.std(all_valid_auc)
    auc_conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_AUC:{all_valid_auc}")
    print(f"MEAN_VALID_AUC:{np.mean(all_valid_auc)}")
    print(f"valid_AUC置信区间:{auc_conf_intveral}")

    # acc
    acc_mean = np.mean(all_valid_acc)
    acc_std = np.std(all_valid_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_ACC:{all_valid_acc}")
    print(f"MEAN_valid_ACC:{np.mean(all_train_acc)}")
    print(f"valid_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_valid_specificity)
    sp_std = np.std(all_valid_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_SP:{all_valid_specificity}")
    print(f"MEAN_VALID_SP:{np.mean(all_train_specificity)}")
    print(f"valid_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_valid_sensitivity)
    se_std = np.std(all_valid_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_SE:{all_valid_sensitivity}")
    print(f"MEAN_VALID_SE:{np.mean(all_valid_sensitivity)}")
    print(f"valid_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_valid_ppv)
    ppv_std = np.std(all_valid_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_ppv:{all_valid_ppv}")
    print(f"MEAN_VALID_ppv:{np.mean(all_valid_ppv)}")
    print(f"valid_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_valid_npv)
    npv_std = np.std(all_valid_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std / np.sqrt(kf.n_splits))
    print(f"ALL_VALID_npv:{all_valid_npv}")
    print(f"MEAN_VALID_npv:{np.mean(all_valid_npv)}")
    print(f"VALID_npv置信区间:{npv_conf_intveral}")

    # 测试集
    print("-" * 40)
    auc_mean = np.mean(all_test_auc)
    auc_std = np.std(all_test_auc)
    conf_intveral = stats.norm.interval(0.95, loc=auc_mean, scale=auc_std / np.sqrt(kf.n_splits))
    print(f"ALL_AUC:{all_test_auc}")
    print(f"MEAN_TEST_AUC:{sum(all_test_auc) / len(all_test_auc)}")
    # print(f"STD_TEST_AUC:{np.std(all_test_auc)}")
    print(f"test_AUC置信区间:{conf_intveral}")

    # acc
    acc_mean = np.mean(all_test_acc)
    acc_std = np.std(all_test_acc)
    acc_conf_intveral = stats.norm.interval(0.95, loc=acc_mean, scale=acc_std / np.sqrt(kf.n_splits))
    print(f"ALL_TEST_ACC:{all_test_acc}")
    print(f"MEAN_TEST_ACC:{np.mean(all_test_acc)}")
    print(f"test_ACC置信区间:{acc_conf_intveral}")

    # SP
    sp_mean = np.mean(all_test_specificity)
    sp_std = np.std(all_test_specificity)
    sp_conf_intveral = stats.norm.interval(0.95, loc=sp_mean, scale=sp_std / np.sqrt(kf.n_splits))
    print(f"ALL_test_SP:{all_test_specificity}")
    print(f"MEAN_TEST_SP:{np.mean(all_test_specificity)}")
    print(f"test_SP置信区间:{sp_conf_intveral}")

    # SE
    se_mean = np.mean(all_test_sensitivity)
    se_std = np.std(all_test_sensitivity)
    se_conf_intveral = stats.norm.interval(0.95, loc=se_mean, scale=se_std / np.sqrt(kf.n_splits))
    print(f"ALL_test_SE:{all_test_sensitivity}")
    print(f"MEAN_TEST_SE:{np.mean(all_test_sensitivity)}")
    print(f"test_SE置信区间:{se_conf_intveral}")

    # PPV
    ppv_mean = np.mean(all_test_ppv)
    ppv_std = np.std(all_test_ppv)
    ppv_conf_intveral = stats.norm.interval(0.95, loc=ppv_mean, scale=ppv_std / np.sqrt(kf.n_splits))
    print(f"ALL_test_ppv:{all_test_ppv}")
    print(f"MEAN_TEST_ppv:{np.mean(all_test_ppv)}")
    print(f"test_ppv置信区间:{ppv_conf_intveral}")

    # NPV
    npv_mean = np.mean(all_test_npv)
    npv_std = np.std(all_test_npv)
    npv_conf_intveral = stats.norm.interval(0.95, loc=npv_mean, scale=npv_std / np.sqrt(kf.n_splits))
    print(f"ALL_test_npv:{all_test_npv}")
    print(f"MEAN_test_npv:{np.mean(all_test_npv)}")
    print(f"test_npv置信区间:{npv_conf_intveral}")
    print("-" * 40)

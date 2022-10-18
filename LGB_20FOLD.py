# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:LGB_20FOLD.py
@Time:2022/7/20 21:26

"""
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.config import *
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')
import joblib


def grid_features(params, train_data, train_label,all_bst_valid_auc):
    train_result, val_result = [], []
    global best_feature_fraction, best_min_child_weight, best_min_child_samples, best_num_leaves, best_max_depth
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2105,
                                                      stratify=train_label)
    best_valid_auc = 0.0
    values_range = list(params.values())
    # 多次网格搜索
    for max_depth in values_range[0]:
        for num_leaves in values_range[1]:
            trn_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            params = {'num_leaves': num_leaves,
                      # 'min_data_in_leaf': 40,
                      'objective': 'binary',
                      'max_depth': max_depth,
                      'learning_rate': 1e-3,
                      # "min_sum_hessian_in_leaf": 4,
                      "boosting": "gbdt",
                      "feature_fraction": 0.7,
                      "bagging_freq": 1,
                      "bagging_fraction": 0.8,
                      "bagging_seed": 11,
                      "lambda_l1": 0.1,
                      # 'lambda_l2': 0.001,
                      "verbosity": -1,
                      "nthread": -1,
                      'metric': {'binary_logloss', 'auc'},
                      "scale_pos_weight": weights,

                      }

            model = lgb.train(params,
                              trn_data,
                              1000,
                              valid_sets=[trn_data, val_data],
                              verbose_eval=False,
                              early_stopping_rounds=40)

            preds_valid = model.predict(X_val, num_iteration=model.best_iteration)
            valid_auc = roc_auc_score(y_score=preds_valid, y_true=y_val)
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_max_depth = max_depth
                best_num_leaves = num_leaves
                joblib.dump(model, "model/lgb_model/lgbm_bst_auc" + str(fold_n) + ".dat")
                print("最优AUC参数：max_depth:%s ,  num_leaves: %s" % (
                    max_depth, num_leaves))
    print("*" * 40)
    print("第二轮参数选择")
    print("*" * 40)
    for min_child_samples in values_range[2]:
        for min_child_weight in values_range[3]:
            for feature_fraction in values_range[4]:
                params["max_depth"] = best_max_depth
                params["num_leaves"] = best_num_leaves
                params["min_child_samples"] = min_child_samples
                params["min_child_weight"] = min_child_weight
                params["feature_fraction"] = feature_fraction
                trn_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                model = lgb.train(params,
                                  trn_data,
                                  1000,
                                  valid_sets=[trn_data, val_data],
                                  verbose_eval=False,
                                  early_stopping_rounds=40)

                preds_valid = model.predict(X_val, num_iteration=model.best_iteration)
                valid_auc = roc_auc_score(y_score=preds_valid, y_true=y_val)
                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    best_min_child_samples = min_child_samples
                    best_min_child_weight = min_child_weight
                    best_feature_fraction = feature_fraction
                    joblib.dump(model, "model/lgb_model/lgbm_bst_auc" + str(fold_n) + ".dat")
                    print("最优AUC参数：min_child_samples: %s , min_child_weight: %s, feature_fraction: %s" % (
                        min_child_samples, min_child_weight, feature_fraction))

    print("*" * 40)
    print("第三轮参数选择")
    print("*" * 40)
    for bagging_fraction in values_range[5]:
        for bagging_freq in values_range[6]:
            for learning_rate in values_range[7]:
                trn_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                params["max_depth"] = best_max_depth
                params["num_leaves"] = best_num_leaves
                params["min_child_samples"] = best_min_child_samples
                params["min_child_weight"] = best_min_child_weight
                params["feature_fraction"] = best_feature_fraction
                params["bagging_fraction"] = bagging_fraction
                params["bagging_freq"] = bagging_freq
                params["learning_rate"] = learning_rate

                model = lgb.train(params,
                                  trn_data,
                                  1000,
                                  valid_sets=[trn_data, val_data],
                                  verbose_eval=False,
                                  early_stopping_rounds=40)

                preds_valid = model.predict(X_val, num_iteration=model.best_iteration)
                valid_auc = roc_auc_score(y_score=preds_valid, y_true=y_val)
                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    # 保存模型
                    joblib.dump(model, "model/lgb_model/lgbm_bst_auc" + str(fold_n) + ".dat")
                    print("最优AUC参数:bagging_fraction:%s , bagging_freq:%s,  learning_rate: %s" % (
                        bagging_fraction, bagging_freq, learning_rate))
    load_new_model = joblib.load("model/lgb_model/lgbm_bst_auc" + str(fold_n) + ".dat")
    # 训练集指标
    preds_train = load_new_model.predict(X_train, num_iteration=load_new_model.best_iteration)
    train_auc = roc_auc_score(y_score=preds_train, y_true=y_train)
    threshold = 0.5
    for pred in preds_train:
        train_result.append(1) if pred > threshold else train_result.append(0)
    # train_acc = accuracy_score(y_true=train_label, y_pred=preds_train)
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
    # 验证集指标
    preds_val = load_new_model.predict(X_val, num_iteration=load_new_model.best_iteration)
    val_auc = roc_auc_score(y_score=preds_val, y_true=y_val)
    # val_acc = accuracy_score(y_true=val_label, y_pred=preds_val)

    threshold = 0.5
    for pred in preds_val:
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
    #判断所有折验证集最优UAC的模型

    if val_auc > all_bst_valid_auc:
        print("&" * 20)
        bst_val_data = np.hstack((X_val, y_val.reshape(-1,1)))
        # np.savetxt("data/bst_val_data.csv",bst_val_data,delimiter=",")
        torch.save(model,"model/lgb_model/all_lgbm_bst_val_auc.pth")
        all_bst_valid_auc = val_auc

        joblib.dump(model, "model/lgb_model/all_lgbm_bst_val_auc.dat")
        model.save_model('model/lgb_model/all_lgbm_bst_val_auc.txt')
        print("当前最优val_auc:",all_bst_valid_auc)
        print("&" * 20)
    return train_auc, train_acc, train_Sp, train_Se, train_ppv, train_npv, val_auc, val_acc, val_Sp, val_Se, val_ppv, val_npv, all_bst_valid_auc


if __name__ == "__main__":
    seed_everything(2022)
    # 八个特征
    data = pd.read_csv("data/seven_feat_221018.csv")

    features = data.iloc[:, 0:-1].astype(float)
    label = data.iloc[:, -1].astype(int)

    # 划分数据集,按照
    features = np.array(features)
    # X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.05, stratify=label)

    # 针对不平衡数据集的权重设置
    weights = len(label[label == 0]) / len(label[label == 1])

    kf = StratifiedKFold(n_splits=20, shuffle=True)

    params = {
        'max_depth': [3, 4, 5,6],  # 指定树的最大深度
        'num_leaves': [6, 15, 31],  # 指定叶子的个数，默认值为31，此参数的数值应该小于2的max_depth次方
        'min_child_samples': [16, 17 ,18,20],  # 叶节点样本的最少数量，默认值20，用于防止过拟合
        'min_child_weight': [1, 3, 5, 6, 7],  # 使一个结点分裂的最小海森值之和
        'feature_fraction': [0.6, 0.7, 0.8, 0.9],  # 特征分数或子特征处理列采样，LightGBM将在每次迭代(树)上随机选择特征子集
        'bagging_fraction': [0.8, 0.9, 1],  # 可以指定每个树构建迭代使用的行数百分比。这意味着将随机选择一些行来匹配每个学习者(树)，（而这里是不放回抽样）
        'bagging_freq': [2, 3,],  # 表示禁用样本采样
        "learning_rate": [0.01, 0.015, 0.03]  # 为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1] 的 eta，设置较小的 eta 就可以多学习几个弱学习器来弥补不足的残差
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
    # 保存最优验证集AUC模型
    # 设置所有折最优AUC：
    bst_val_auc = 0.0

    for fold_n, (train_index, test_index) in enumerate(kf.split(features, label)):
        test_result = []
        print("fold {}".format(fold_n + 1))
        features = np.array(features)

        X_train, y_train = features[train_index], np.array(label)[train_index].astype(int)
        X_test, y_test = features[test_index], np.array(label)[test_index].astype(int)

        tr_auc, tr_acc, tr_Sp, tr_Se, tr_ppv, tr_npv, \
        valid_auc, valid_acc, valid_Sp, valid_Se, valid_ppv, valid_npv,all_bst_valid_auc= grid_features(params, X_train, y_train,bst_val_auc)
        bst_val_auc = all_bst_valid_auc
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

        load_model = joblib.load("model/lgb_model/lgbm_bst_auc" + str(fold_n) + ".dat")
        preds_test = load_model.predict(X_test, num_iteration=load_model.best_iteration)
        fpr, tpr, thresholds = roc_curve(y_test, preds_test)

        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
        test_auc = roc_auc_score(y_score=preds_test, y_true=y_test)

        # test_acc = accuracy_score(y_true=y_test, y_pred=preds_test)
        threshold = 0.5
        for pred in preds_test:
            test_result.append(1) if pred > threshold else test_result.append(0)
        confusion = confusion_matrix(y_true=y_test, y_pred=test_result)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        test_acc = round((TP + TN) / (TP + FP + FN + TN), 3)
        # "Sensitivity: "
        test_Se = round(TP / (TP + FN+0.01), 3)
        # "Specificity: "
        test_Sp = round(1 - (FP / (FP + TN+0.01)), 3)
        # "Positive predictive value: "
        test_ppv = round(TP / (TP + FP+0.01), 3)
        # "Negative predictive value: "
        test_npv = round(TN / (FN + TN+0.01), 3)
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
    np.save("model/lgb_model/lgb.npy", arr)
    print("save npy done")

    # 计算置信区间
    # auc
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

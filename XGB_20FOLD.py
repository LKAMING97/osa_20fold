# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:XGB_20FOLD.py
@Time:2022/7/20 14:34

"""

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, auc
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from core.config import *
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
# windows解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
import warnings
import joblib

warnings.filterwarnings('ignore')


def grid_features(params, train_data, train_label):

    train_result, val_result = [], []
    global bst_colsample_bytree, bst_subsample, bst_max_depth
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2105,
                                                                    stratify=train_label)
    best_valid_auc = 0.0
    values_range = list(params.values())
    # 多次网格搜索
    for max_depth in values_range[0]:
        for subsample in values_range[1]:
            for colsample_bytree in values_range[2]:
                trn_data = xgb.DMatrix(train_data, label=train_label)
                valid_data = xgb.DMatrix(val_data, label=val_label)
                watchlist = [(trn_data, 'train'), (valid_data, 'valid')]
                print("本次所使用参数：max_depth:%s , subsample:%s,  colsample_bytree: %s" % (
                    max_depth, subsample, colsample_bytree))

                params = {
                    'booster': 'gbtree',
                    'objective': 'binary:logistic',  # 目标函数，可自定义,binary:logistic输出不是0,1而是小数，后续输出以0.5为阈值进行分类
                    'eval_metric': ["logloss", "auc"],  # 评级估量
                    'gamma': 0.1,  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值
                    'max_depth': max_depth,  # 树的深度
                    'alpha': 1,  # L1正则系数
                    'lambda': 1,  # L2正则系数
                    'subsample': subsample,  # 训练每棵树时，使用数据占全部训练集比重
                    'colsample_bytree': colsample_bytree,  # 随机使用多少特征生成决策树
                    'min_child_weight': 1,  # 在子节点中实例权重的最小的和；
                    'eta': 1e-2,  # lr
                    'nthread': -1,  # 使用全部CPU进行并行运算（默认）
                    'seed': 2022,
                    "tree_method": "gpu_hist",
                    "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # 针对不平衡标签
                }

                bst = xgb.train(params, trn_data, 1000, watchlist, early_stopping_rounds=40, verbose_eval=False)

                preds_valid = bst.predict(xgb.DMatrix(val_data), ntree_limit=bst.best_ntree_limit, )
                valid_auc = roc_auc_score(y_score=preds_valid, y_true=val_label)
                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    bst_max_depth = max_depth
                    bst_subsample = subsample
                    bst_colsample_bytree = colsample_bytree
                    # params["colsample_bytree"] = colsample_bytree
                    # 保存模型
                    joblib.dump(bst, "model/xgb_model/xgboost_bst_auc" + str(fold_n) + ".dat")
                    print("最优AUC参数：max_depth:%s ,  subsample: %s, colsample_bytree: %s" % (
                        max_depth, subsample, colsample_bytree))
    print("*" * 40)
    print("第二轮参数选择")
    print("*" * 40)
    for min_child_weight in values_range[3]:
        for gamma in values_range[4]:
            for eta in values_range[5]:
                trn_data = xgb.DMatrix(train_data, label=train_label)
                valid_data = xgb.DMatrix(val_data, label=val_label)
                watchlist = [(trn_data, 'train'), (valid_data, 'valid')]
                print("本次所使用参数：min_child_weight:%s , gamma:%s,  eta: %s" % (
                    min_child_weight, gamma, eta))
                params["max_depth"] = bst_max_depth
                params["subsample"] = bst_subsample
                params["colsample_bytree"] = bst_colsample_bytree
                params["min_child_weight"] = min_child_weight
                params["gamma"] = gamma
                params["eta"] = eta

                bst = xgb.train(params, trn_data, 1000, watchlist, early_stopping_rounds=40, verbose_eval=False)
                preds_valid = bst.predict(xgb.DMatrix(val_data), ntree_limit=bst.best_ntree_limit, )
                valid_auc = roc_auc_score(y_score=preds_valid, y_true=val_label)
                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    # 保存模型
                    joblib.dump(bst, "model/xgb_model/xgboost_bst_auc" + str(fold_n) + ".dat")
                    print("最优AUC:参数min_child_weight:%s , gamma:%s,  eta: %s" % (
                        min_child_weight, gamma, eta))
    final_model = joblib.load("model/xgb_model/xgboost_bst_auc" + str(fold_n) + ".dat")
    new_config = final_model.save_config()
    print(new_config)
    load_new_model = joblib.load("model/xgb_model/xgboost_bst_auc" + str(fold_n) + ".dat")
    # 训练集指标
    preds_train = load_new_model.predict(xgb.DMatrix(train_data), ntree_limit=load_new_model.best_ntree_limit)
    train_auc = roc_auc_score(y_score=preds_train, y_true=train_label)
    threshold = 0.5
    for pred in preds_train:
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
    preds_val = load_new_model.predict(xgb.DMatrix(val_data), ntree_limit=load_new_model.best_ntree_limit)
    val_auc = roc_auc_score(y_score=preds_val, y_true=val_label)

    # val_acc = accuracy_score(y_true=val_label, y_pred=preds_val)
    threshold = 0.5
    for pred in preds_val:
        val_result.append(1) if pred > threshold else val_result.append(0)
    confusion = confusion_matrix(y_true=val_label, y_pred=val_result)
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
    seed_everything(2022)
    # 八个特征
    data = pd.read_csv("data/test_data.csv")

    features = data.iloc[:, 0:-1].astype(float)
    label = data.iloc[:, -1].astype(int)

    # 划分数据集,按照
    features = np.array(features)
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.05, stratify=label)

    kf = StratifiedKFold(n_splits=20, shuffle=True)

    # 针对不平衡数据集的权重设置
    weights = len(label[label == 0]) / len(label[label == 1])
    # 搜索参数，设置不同组参数进行网格搜索,选择最优
    params = {
        'max_depth': [3, 4, 5],  # 树的深度
        'subsample': [0.5, 0.6, 0.7],  # 样本采样率
        "colsample_bytree": [0.6, 0.7, 0.8],  # 特征列所对应的特征值采样率
        "min_child_weight": [1, 2],  # 子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
        "gamma": [0.1, 0.2],  # 就是 从GBDT到XGBoost 中的正则化项控制叶子节点数量复杂度
        "eta": [1e-2, 2e-2]  # 学习率
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
    feature_importance_df = pd.DataFrame()
    # 设置不同组参数进行网格搜索,选择最优
    bst_val_auc = 0.0
    for fold_n, (train_index, test_index) in enumerate(kf.split(features, label)):
        print('Outer Fold-', fold_n + 1)
        test_result = []
        # 划分训练集和测试集
        X_train, y_train = features[train_index], np.array(label)[train_index].astype(int)
        X_test, y_test = features[test_index], np.array(label)[test_index].astype(int)

        tr_auc, tr_acc, tr_Sp, tr_Se, tr_ppv, tr_npv, \
        valid_auc, valid_acc, valid_Sp, valid_Se, valid_ppv, valid_npv = grid_features(params, X_train, y_train)
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
        load_model = joblib.load("model/xgb_model/xgboost_bst_auc" + str(fold_n) + ".dat")

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = load_model.get_fscore().keys()
        fold_importance_df["importance"] = load_model.get_fscore().values()
        fold_importance_df["fold"] = fold_n + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # # 预测训练集
        # preds_train = load_model.predict(xgb.DMatrix(train_data), ntree_limit=load_model.best_ntree_limit)
        # fpr, tpr, thresholds = roc_curve(X_train, preds_train)
        # train_auc = roc_auc_score(y_score=preds_train, y_true=y_train)

        # 预测测试集
        preds_test = load_model.predict(xgb.DMatrix(X_test), ntree_limit=load_model.best_ntree_limit)
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
        test_Se = round(TP / (TP + FN), 3)
        # "Specificity: "
        test_Sp = round(1 - (FP / (FP + TN)), 3)
        # "Positive predictive value: "
        test_ppv = round(TP / (TP + FP), 3)
        # "Negative predictive value: "
        test_npv = round(TN / (FN + TN), 3)
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
    arr =list()
    arr.append(mean_tpr)
    arr.append(mean_fpr)
    arr.append(mean_auc)
    np.save("model/xgb_model/xgb.npy", arr)
    print("save npy done")
    # plt.plot(mean_fpr, mean_tpr, 'k--', label='XGB ROC (area = {0:.2f})'.format(mean_auc), lw=2)
    #
    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # # plt.show()
    # plt.close()
    ## plot feature importance
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
            :5].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',
                                                                                                    ascending=False)
    plt.figure(figsize=(8, 10))
    sns.barplot(y="Feature",
                x="importance",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('model/xgb_model/xgb_importances.png')
    # 计算各项指标,注意置信区间传递的是标准误差
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



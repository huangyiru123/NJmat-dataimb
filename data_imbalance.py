import argparse
import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import pickle

def data_imbalance_randomforest_classifier(num_1, num_2, num_3, num_4, num_5, num_6, path, csvname):
    # 读取数据
    data = pd.read_csv(csvname)
    X = data.values[:, 1:-1]
    y = data.values[:, -1]

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # 数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 处理数据不平衡问题
    smote = SMOTE(random_state=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 网格搜索参数设置
    param_grid = {
        'n_estimators': [50, 80, 100, 120],
        'max_depth': [6, 7],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1, 2],
        'random_state': [0, 1]
    }

    # 创建随机森林分类器
    rfc = RandomForestClassifier()

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_res, y_train_res)

    # 使用最佳参数训练模型
    best_params = grid_search.best_params_
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train_res, y_train_res)

    # 保存模型
    pickle.dump(clf, open(os.path.join(path, "Classified_two_RF.dat"), "wb"))

    # 画出ROC曲线和混淆矩阵
    def plot_roc_curve(clf, X, y, path, filename):
        y_score = clf.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(path, filename), dpi=300)
        plt.close()

    def plot_confusion_matrix(clf, X, y, path, filename):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = range(len(set(y)))
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(path, filename), dpi=300)
        plt.close()

    plot_roc_curve(clf, X_test, y_test, path, 'RandomForest_test_ROC.png')
    plot_confusion_matrix(clf, X_test, y_test, path, 'RandomForest_test_CM.png')
    plot_roc_curve(clf, X_train_res, y_train_res, path, 'RandomForest_train_ROC.png')
    plot_confusion_matrix(clf, X_train_res, y_train_res, path, 'RandomForest_train_CM.png')

    return f"Best parameters: {best_params}", f"ROC and Confusion Matrix saved in {path}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_1", type=int, required=True)
    parser.add_argument("--num_2", type=int, required=True)
    parser.add_argument("--num_3", type=int, required=True)
    parser.add_argument("--num_4", type=int, required=True)
    parser.add_argument("--num_5", type=int, required=True)
    parser.add_argument("--num_6", type=int, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--csvname", type=str, required=True)
    args = parser.parse_args()

    result = data_imbalance_randomforest_classifier(
        args.num_1, args.num_2, args.num_3, args.num_4, args.num_5, args.num_6, args.path, args.csvname)
    print(result)

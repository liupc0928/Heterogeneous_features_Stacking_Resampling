import numpy as np
import pandas as pd

from Features.HuMoments import hu_monments as Hu
from Features.GraphFeats import Graph
from Features.StructureInFeats import Tensor, LBP

from Heterogeneous.utils import TransferLabel, plot_confusion_matrix as pltcm, tsne, normalization as norm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from xgboost import XGBClassifier as XGB, plot_importance
from lightgbm import LGBMClassifier as LGBM
from sklearn.linear_model import LogisticRegression as Logist
import math

import warnings

warnings.filterwarnings('ignore')

LE = LabelEncoder()
mms = MinMaxScaler()
col = ['CAL', 'GR', 'SP', 'LLD', 'LLS', 'AC', 'DEN']
L = ['泥岩', '粉砂岩', '泥质粉砂岩', '粉砂质泥岩']


def Feat(data):
    r1, r2, r3 = 10, 10, 10
    tens = Tensor(data, r1)
    lbp = LBP(data, r2)
    hu = Hu(data, r3)
    g = Graph(data)
    X1, X2 = g.GFeats()

    return data, np.hstack((tens, lbp, hu)), np.hstack((X1, X2))


def RFrun(x_train, x_test, y_train):
    rf = RF(max_depth=8, oob_score=True, random_state=10, n_jobs=-1)
    rf.fit(x_train, y_train)
    return rf.predict_proba(x_test)


def GBDTrun(x_train, x_test, y_train):
    ytrain = LE.fit_transform(y_train)
    gbdt = GBDT(max_depth=8, learning_rate=0.1, random_state=10)
    gbdt.fit(x_train, ytrain)
    return gbdt.predict_proba(x_test)


def XGBrun(x_train, x_test, y_train):
    ytrain = LE.fit_transform(y_train)
    xgb = XGB(max_depth=5, learning_rate=0.1, random_state=10, n_jobs=-1)
    xgb.fit(x_train, ytrain)
    return xgb.predict_proba(x_test)


def LGBMrun(x_train, x_test, y_train):
    lgbm = LGBM(max_depth=6, num_leaves=15, objective='multiclassova', learning_rate=0.1, n_jobs=-1)
    lgbm.fit(x_train, y_train)
    return lgbm.predict_proba(x_test)


def metrics(y_test, y_pre, title=None):
    ACC = round(accuracy_score(y_test, y_pre) * 100, 2)
    F1 = round(f1_score(y_test, y_pre, average='macro') * 100, 2)

    cm = confusion_matrix(y_test, y_pre)
    # pltcm(cm, ld.classes(), title=title, normalize=True)  # normalize=True 绘制百分比

    # print(cm)
    # print(classification_report(y_test, y_pre, digits=4))
    return ACC, F1


def BaseLearners(trainD, testD):
    x_train, y_train = trainD.drop(columns=['Liths']), trainD['Liths']
    x_test, y_test = testD.drop(columns=['Liths']), testD['Liths']
    mms.fit(x_train)
    x_train = mms.transform(x_train)
    x_test = mms.transform(x_test)

    p1 = RFrun(x_train, x_test, y_train)
    p2 = GBDTrun(x_train, x_test, y_train)
    p3 = XGBrun(x_train, x_test, y_train)
    p4 = LGBMrun(x_train, x_test, y_train)

    return np.hstack((p1, p2, p3, p4))


def resampling(Data):
    M = Data[Data["Liths"] == "泥岩"].reset_index(drop=True)
    S = Data[Data["Liths"] == "粉砂岩"].reset_index(drop=True)
    SM = Data[Data["Liths"] == "粉砂质泥岩"].reset_index(drop=True)
    AS = Data[Data["Liths"] == "泥质粉砂岩"].reset_index(drop=True)

    data = M
    n = len(M) / len(AS)
    n = int(n) if round(math.modf(n)[0], 1) < 0.3 else math.ceil(n)
    Len = int(len(M) / n)
    index = [i for i in range(0, len(data), Len)]
    resampled_M = []
    for i in range(1, len(index) - 1):
        resampled_M.append(data.iloc[index[i - 1]:index[i]])
    resampled_M.append(data.iloc[index[-2]:len(data)])

    data = S
    n = len(S) / len(AS)
    n = int(n) if round(math.modf(n)[0], 1) < 0.3 else math.ceil(n)
    Len = int(len(S) / n)
    index = [i for i in range(0, len(data), Len)]
    resampled_S = []
    for i in range(1, len(index)):
        resampled_S.append(data.iloc[index[i - 1]:index[i]])
    resampled_S.append(data.iloc[index[-2]:len(data)])

    SM1 = SM.iloc[:len(AS)]
    SM2 = SM.iloc[-len(AS):]

    res_X = []
    for i in [SM1, SM2]:
        for j in resampled_S:
            for k in resampled_M:
                res_X.append(pd.concat([AS, i, j, k]))

    return res_X


def Meta_Learning(trainD, testD):
    X_resampled = resampling(trainD)
    P = []
    for S in X_resampled:
        P.append(BaseLearners(S, testD))
    X = np.hstack(tuple(P))

    MetaLearner = Logist(random_state=10)
    MetaLearner.fit(X, testD['Liths'])
    predict = MetaLearner.predict(X)
    ACC, F1 = metrics(testD['Liths'], predict)
    print('\nMeta: ACC {}, F1 {}'.format(ACC, F1))

    return predict


def BaseLines(trainD, testD):
    x_train, y_train = trainD.drop(columns=['Liths']), trainD['Liths']
    x_test, y_test = testD.drop(columns=['Liths']), testD['Liths']
    mms.fit(x_train)
    x_train = mms.transform(x_train)
    x_test = mms.transform(x_test)

    rf = RF(max_depth=8, oob_score=True, random_state=10, n_jobs=-1)
    rf.fit(x_train, y_train)
    ACC, F1 = metrics(y_test, rf.predict(x_test))
    print('\nRF: ACC {}, F1 {}'.format(ACC, F1))

    LE = LabelEncoder()
    ytrain = LE.fit_transform(y_train)
    gbdt = GBDT(max_depth=8, learning_rate=0.1, random_state=10)  # [5, 15]
    gbdt.fit(x_train, ytrain)
    ACC, F1 = metrics(y_test, LE.inverse_transform(gbdt.predict(x_test)))
    print('\nGBDT: ACC {}, F1 {}'.format(ACC, F1))

    xgb = XGB(max_depth=5, learning_rate=0.1, random_state=10, n_jobs=-1)
    xgb.fit(x_train, ytrain)
    ACC, F1 = metrics(y_test, LE.inverse_transform(xgb.predict(x_test)))
    print('\nXGBoost: ACC {}, F1 {}'.format(ACC, F1))

    lgbm = LGBM(max_depth=6, num_leaves=15, objective='multiclassova', learning_rate=0.1, n_jobs=-1)
    lgbm.fit(x_train, y_train)
    lgbm.fit(x_train, y_train)
    ACC, F1 = metrics(y_test, lgbm.predict(x_test))
    print('\nLightGBM: ACC {}, F1 {}'.format(ACC, F1))

    return LE.inverse_transform(gbdt.predict(x_test)), LE.inverse_transform(
        xgb.predict(x_test)), rf.predict(x_test), lgbm.predict(x_test)


if __name__ == '__main__':

    def save_csv(trainD, trainF, TD, TF, ytest, name):
        Ogbdt, Oxgb, Orf, Olgbm = BaseLines(trainD, TD)
        Ometa = Meta_Learning(trainD, TD)
        Fmeta = Meta_Learning(trainF, TF)

        T = pd.DataFrame()
        T['Liths'] = ytest
        T['GBDT'] = Ogbdt
        T['XGBoost'] = Oxgb
        T['RF'] = Orf
        T['LightGBM'] = Olgbm
        T['Meta'] = Ometa
        T['Meta_features'] = Fmeta

        # T.to_csv('../save/' + name + '.csv', index=False)


    source = ['A1', 'A2']
    target = ['B1', 'B2', 'B3', 'B4']

    W = []
    for n in source + target:
        w = pd.read_csv('WellData/' + n + '.csv', encoding='gbk')

        w = w.query('GR<290')
        w = w.query('LLD<100')
        w = w.query('LLS<100')
        w = w.reset_index(drop=True)

        w = w[col + ['Liths']]
        W.append(w)

    """orignal"""
    WO = []
    for wo in W:
        WO.append(wo[wo['Liths'].isin(L)].reset_index(drop=True))
    [SO1, SO2, TO1, TO2, TO3, TO4] = WO

    """heterogeneous"""
    WF = []
    for w in W:
        data, F, G = Feat(w[col])
        wf = np.hstack((data, F, G))
        wf = pd.concat([pd.DataFrame(wf), w['Liths']], axis=1)
        WF.append(wf[wf['Liths'].isin(L)].reset_index(drop=True))
    [SF1, SF2, TF1, TF2, TF3, TF4] = WF

    trainD = pd.concat([SO1, SO2]).reset_index(drop=True)
    trainF = pd.concat([SF1, SF2]).reset_index(drop=True)

    print('\n\nB1')
    save_csv(trainD, trainF, TO1, TF1, TO1['Liths'], "B1")

    print('\n\nB2')
    save_csv(trainD, trainF, TO2, TF2, TO2['Liths'], "B2")

    print('\n\nB3')
    save_csv(trainD, trainF, TO3, TF3, TO3['Liths'], "B3")

    print('\n\nB4')
    save_csv(trainD, trainF, TO4, TF4, TO4['Liths'], "B4")

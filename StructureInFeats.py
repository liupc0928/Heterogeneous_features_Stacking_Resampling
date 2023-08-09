import pandas as pd
import numpy as np
from scipy.signal import find_peaks
np.set_printoptions(threshold=np.inf)


def LBP(data, step):
    def LBP_Sig(InSig, step):

        n = len(InSig)
        # 首先提取局部梯度角度信息
        TanSig = np.tan(InSig / step)
        # 提取峰值
        locs, properties = find_peaks(InSig)
        # 提取混合局部LBP特征
        mV = np.mean(InSig)
        mdV = np.median(InSig)
        Mlbp = np.zeros((n, 1))
        # MLBP编码对比向量：mV, mdV, leftPk, rightPk, TanL2, TanL1, TanR1, TanR2
        Temp = [[128], [64], [32], [16], [8], [4], [2], [1]]
        Temp = np.asarray(Temp)
        for k in range(2, n - 2):
            codes = np.zeros((8, 1))
            tV = InSig[k]
            tTan = TanSig[k]
            if tV < mV:
                codes[0] = 1
            elif tV == mV:
                codes[0] = 1 / 2

            if tV < mdV:
                codes[1] = 1
            elif tV == mdV:
                codes[1] = 1 / 2

            tind = [x for x in locs if x != 0 and x < k]
            if len(tind) != 0:
                t = tind[-1]
                if tV < InSig[t]:
                    codes[2] = 1
                elif tV == InSig[t]:
                    codes[2] = 1 / 2

            tind = [x for x in locs if x >= k]
            if len(tind) >= 2:
                t = tind[0:2]
                if tV < InSig[t[1]]:
                    codes[3] = 1
                elif tV == InSig[t[1]]:
                    codes[3] = 1 / 2

            if tTan < TanSig[k - 2]:
                codes[4] = 1
            elif tTan == TanSig[k - 2]:
                codes[4] = 1 / 2

            if tTan < TanSig[k - 1]:
                codes[5] = 1
            elif tTan == TanSig[k - 1]:
                codes[5] = 1 / 2

            if tTan < TanSig[k + 1]:
                codes[6] = 1
            elif tTan == TanSig[k + 1]:
                codes[6] = 1 / 2

            if tTan < TanSig[k + 2]:
                codes[7] = 1
            elif tTan == TanSig[k + 2]:
                codes[7] = 1 / 2

            Mlbp[k] = sum(codes * Temp)

        Mlbp[:2] = Mlbp[2]
        Mlbp[n - 2:n] = Mlbp[n - 3]

        return Mlbp

    # 多曲线
    # data = norm(data)
    data = np.array(data)
    m, n = data.shape
    Mlbp = np.zeros((m, n))
    for j in range(n):
        Mlbp[:, j] = LBP_Sig(data[:, j], step)[:, 0]

    return Mlbp


def Tensor(data, radius):

    # data = norm(data)
    data = np.array(data)
    m, n = data.shape

    def Sig(curve):

        curve = np.array(curve).reshape(m, 1)
        tensor = np.zeros((m, 1))
        for k in range(radius, m - radius):
            tblock = curve[k - radius:k + radius + 1]
            _, v, _ = np.linalg.svd(tblock)
            tensor[k] = v

        tensor[:radius] = tensor[radius]
        tensor[m - radius: m] = tensor[m - radius - 1]

        return tensor

    tensors = np.zeros((m, n))
    for i in range(n):
        t = Sig(data[:, i])
        tensors[:, i] = t[:, 0]

    return tensors

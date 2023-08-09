import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.manifold import TSNE
import seaborn as sns
import itertools
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def split(x, y):
    _, _, _, _, train_i, test_i = train_test_split(x, y, np.arange(len(y)), test_size=0.3, random_state=5, stratify=y)
    y_train = y[train_i]
    y_test = y[test_i]

    return y_train, y_test, train_i, test_i


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, modeln=None, title=None, cmap=plt.cm.Greens):
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    fig = plt.figure(figsize=(3.3, 3.3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = plt.colorbar(fraction=0.046, pad=0.05)
    cb.set_ticks(range(0, 100, 20))

    # font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 7.5}
    font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 8}
    # font = {'family': 'serif', 'serif': 'Arial', 'weight': 'normal', 'size': 12}
    plt.rc('font', **font)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    if 'original' in str(title):
        plt.xlabel('Real label\n(a)')  # 原始
    elif 'integrated' in str(title):
        plt.xlabel('Real label\n(b)')  # 特征
    else:
        plt.xlabel('Real label')
    plt.ylabel('Predicted label')
    plt.title(title)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 # color='black'
                 )
    plt.tight_layout()

    # plt.savefig(r'C:\Users\DELL\Desktop\论文图\Times/' + title + '.tiff', transparent=True, dpi=300)

    plt.show()


# 降维可视化
def tsne(data, labels, name=None):

    font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 11}
    plt.rc('font', **font)

    tsne = TSNE(n_components=2, init='pca', random_state=42)
    reduced_data = tsne.fit_transform(data)
    reduced_data = np.transpose(reduced_data)

    class_num = len(np.unique(labels))
    df = pd.DataFrame()
    # df["y"] = labels[0]
    df["y"] = labels
    df["comp-1"] = reduced_data[0]
    df["comp-2"] = reduced_data[1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df).set(title=name)

    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig('C:/Users/DELL/Desktop/new/' + name + '.tiff', dpi=300)

    plt.show()


# 归一化
def normalization(data):
    norm = MinMaxScaler()
    return norm.fit_transform(data)


# 转换标签
def TransferLabel(Liths):
    LE = LabelEncoder()
    label = LE.fit_transform(Liths)
    liths_map = LE.inverse_transform(list(set(label)))

    return label, liths_map

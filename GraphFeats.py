import numpy as np
import pandas as pd
import networkx as nx


class Graph():
    def __init__(self, data):
        self.data = data
        self.norData = np.array(self.data)

    def laplacian(self, adj_matrix):
        # 先求度矩阵
        R = np.sum(adj_matrix, axis=1)
        degreeMatrix = np.diag(R)
        return degreeMatrix - adj_matrix

    def decomposition(self, g):
        A = np.array(nx.adjacency_matrix(g).todense())
        LA = self.laplacian(A)

        eig_values, eig_vectors = np.linalg.eig(LA)
        return eig_values

    def CorrG(self):
        data = pd.DataFrame(self.norData, columns=self.data.columns)
        m, n = data.shape
        X = np.zeros((m, n))
        for i in range(m):
            E = []
            for c1 in data.columns:
                for c2 in data.columns:
                    if c1 != c2:
                        w = np.linalg.norm(data.iloc[i][c1] - data.iloc[i][c2])
                        E.append((c1, c2, w))
            g = nx.Graph()
            g.add_weighted_edges_from(E)
            X[i] = self.decomposition(g)
        return X

    def GldInf(self):
        data = self.norData
        Mea = np.mean(data, axis=0)
        Med = np.median(data, axis=0)
        sd = list(np.sum(np.abs(data), axis=1))
        Max = data[sd.index(max(sd))]
        Min = data[sd.index(min(sd))]

        return Max, Min, Mea, Med

    def LocalG(self):
        Max, Min, Mea, Med = self.GldInf()
        X = self.norData
        m, n = X.shape

        X2 = np.zeros((m, 7))
        for i in range(m):
            current = i
            previous = i - 1
            next = i + 1
            if current == 0:
                previous = i + 2
            if current == m - 1:
                next = i - 2
            xi = np.array([X[current], X[previous], X[next], Max, Min, Mea, Med])

            E = []
            for node in range(1, xi.shape[0]):
                w = np.linalg.norm(xi[0] - xi[node])
                E.append((0, node, w))

            g = nx.DiGraph()
            g.add_weighted_edges_from(E)
            X2[i] = self.decomposition(g)
        return X2

    def GFeats(self):
        X1 = self.CorrG()
        X2 = self.LocalG()

        return X1, X2

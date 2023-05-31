import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pygmo import hypervolume
from pymoo.indicators.hv import HV
import os

np.random.seed(7)


class ObjectivesSpace:
    def __init__(self, df, functions):
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        # pts = obj_pts.copy()
        # obj_pts = obj_pts[obj_pts.sum(1).argsort()[::-1]]
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return pd.DataFrame(self.points[0], columns=self._constr_obj())

    def get_dominated(self):
        return pd.DataFrame(self.points[1], columns=self._constr_obj())


    def _min_max(self, X):
        M = X.copy()
        max = M.max(axis=0)
        min = M.min(axis=0)
        M = (M - min) / (max - min)
        return M


    def _compute_hypervolume(self, x, r):
        x = x[list(self.functions.keys())]
        ind = HV(ref_point=r)
        return ind(np.array(x))

    def hypervolumes(self, r):
        factors = np.array(list(map(lambda x: -1 if x == 'max' else 1, list(self.functions.values()))))
        hv_pts = np.copy(self.points[0])
        hv_pts[:, 1:] = hv_pts[:, 1:] * factors
        r = r * factors
        not_dominated = pd.DataFrame(hv_pts, columns=self._constr_obj())
        not_dominated['hypervolume'] = \
            not_dominated.apply(lambda x: self._compute_hypervolume(x, r), axis=1)
        return not_dominated

    def knee_point(self, scale=False):
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        kp_pts = np.copy(self.points[0])
        if scale:
            kp_pts[:, 1:] = self._min_max(kp_pts[:, 1:])
        kp_pts[:, 1:] = kp_pts[:, 1:] * factors
        counter = np.zeros(kp_pts[:, 1:].shape[0], dtype=np.int64)
        for i in range(1000):
            w = np.random.uniform(low=0, high=1, size=kp_pts[:, 1:].shape)
            for j in range(0, w.shape[0]):
                w[j] = w[j] / w[j].sum()
            counter[np.argmax((kp_pts[:, 1:] * w).sum(axis=1))] = counter[np.argmax((kp_pts[:, 1:] * w).sum(axis=1))] + 1
        columns = self._constr_obj()
        columns.append('utility_based')
        return pd.DataFrame(np.column_stack((kp_pts, counter)), columns=columns)

    def euclidean_distance(self, up, scale=False):
        ed_points = np.copy(self.points[0])
        if scale:
            ed_points[:, 1:] = self._min_max(ed_points[:, 1:])
        ed_points[:, 1:] = (ed_points[:, 1:] - up) ** 2
        distances = ed_points[:, 1:].sum(axis=1) ** (1 / 2)
        columns = self._constr_obj()
        columns.append('euclidean_distance')
        return pd.DataFrame(np.column_stack((ed_points, distances)), columns=columns)

    def weighted_mean(self, w, scale=False):
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        wm_points = np.copy(self.points[0])
        if scale:
            wm_points[:, 1:] = self._min_max(wm_points[:, 1:])
        wm_points[:, 1:] = wm_points[:, 1:] * factors
        wm_points[:, 1:] = wm_points[:, 1:] * w
        sum = wm_points[:, 1:].sum(axis=1) / wm_points[:, 1:].shape[1]
        columns = self._constr_obj()
        columns.append('weighted_mean')
        return pd.DataFrame(np.column_stack((wm_points, sum)), columns=columns)

    def _compute_distances(self, model, up, scale, calibration, dataset):
        relative_path = f'data/population/{dataset}'
        dir = os.listdir(relative_path)
        for el in dir:
            if model in el:
                model_per_user = el
                break
            else:
                model_per_user = ''
        df = pd.read_csv(relative_path + '/' + model_per_user, sep='\t')
        if calibration:
            df = pd.merge(df, up, on=['User'])
            M = df[list(self.functions.keys())].values
            up = df[[s + '_up' for s in list(self.functions.keys())]].values
        else:
            M = df[list(self.functions.keys())].values
        if scale:
            M = self._min_max(M)
        M = (up - M) ** 2
        M = M.sum(axis=1)
        return np.log(M.sum())

    def _variance(self, distances):
        mean = distances.mean()
        variance = ((distances - mean) ** 2).sum() / distances.shape[0]
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def pdu(self, up, dataset, scale=False, calibration=False):
        not_dominated = pd.DataFrame(self.points[0], columns=self._constr_obj())
        if calibration:
            not_dominated['c_pdu'] = not_dominated['model'].map(
                lambda x: self._compute_distances(x, up, scale, calibration, dataset))
        else:
            not_dominated['pdu'] = not_dominated['model'].map(
                lambda x: self._compute_distances(x, up, scale, calibration, dataset))
        return not_dominated


    #def plot(self, not_dominated, dominated, r):
    #    not_dominated = not_dominated.values
    #    dominated = dominated.values
    #    fig = plt.figure()
    #    if not_dominated.shape[1] == 3:
    #        ax = fig.add_subplot()
    #        ax.scatter(dominated[:, 1], dominated[:, 2], color='red')
    #        ax.scatter(not_dominated[:, 1], not_dominated[:, 2], color='blue')
    #        ax.scatter(r[0], r[1], color='green')
    #        plt.xlim(not_dominated[:, 1].min(), not_dominated[:, 1].max())
    #        plt.ylim(not_dominated[:, 2].min(), not_dominated[:, 2].max())
    #        plt.show()
    #    elif not_dominated.shape[1] == 4:
    #        ax = fig.add_subplot(projection='3d')
    #        ax.scatter(dominated[:, 1], dominated[:, 2], dominated[:, 3], color='red')
    #        ax.scatter(not_dominated[:, 1], not_dominated[:, 2], not_dominated[:, 3], color='blue')
    #        ax.scatter(r[0], r[1], r[2], color='green')
    #        ax.set_xlim3d(not_dominated[:, 1].min(), not_dominated[:, 1].max())
    #        ax.set_ylim3d(not_dominated[:, 2].min(), not_dominated[:, 2].max())
    #        ax.set_zlim3d(not_dominated[:, 3].min(), not_dominated[:, 3].max())
    #        plt.show()
    #    else:
    #        print("Cannot print >3-dimensional objective funtion space")


        # ax.plot(*not_dominated[not_dominated[:, 2].argsort()].T)
        #for i, txt in enumerate(not_dominated[:, 3]):
        #    ax.annotate(str(txt), (not_dominated[:, 1][i] * 1.005, not_dominated[:, 2][i]), fontsize=5)
        #for i, txt in enumerate(dominated[:, 3]):
        #    ax.annotate(str(txt), (dominated[:, 1][i] * 1.005, dominated[:, 2][i]), fontsize=5)
        # plt.axis([0, 1, 0, 1])



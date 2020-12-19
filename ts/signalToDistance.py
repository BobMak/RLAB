# import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib as mpl
import matplotlib.pyplot as plt

from ToPy import get_rssi_locations, get_single_gw_rssi_locations
from utils import cache


PIXEL_SCALE = 41.5687


class BaseRSSIModel:
    def predict(self, rssi):
        return [np.exp((rssi[0][0]+55.97607)/-11.27108)*PIXEL_SCALE]


if __name__ == "__main__":
    # Get the data. Will create huge cache files on first run
    print('---loading a dataset---')
    # timestamp, rssi, sr_x, sr_y, gw_x, gw_y = cache('../datasets/rssi_cache.pkl', get_rssi_locations,
    #                                                 ["../datasets/sensors"])
    for g in (0,4,6,7):
        dist = []
        test_rssi = []
        test_dist = []
        timestamp, rssi, sr_x, sr_y, gw_x, gw_y = cache(f'../datasets/rssi_singlegw_cache-{g}.pkl', get_single_gw_rssi_locations,
                                                        ("../datasets/sensors", g))
        locs = {}
        locs_test = []
        for sx, sy, gx, gy, r in zip(sr_x, sr_y, gw_x, gw_y, rssi):
            d = np.sqrt((sx - gx)**2 + (sy - gy)**2)
            dist.append(d)
            if d not in locs:
                locs[d] = [r]
            else:
                locs[d].append(r)
        # testing set should have 10% of unique locations
        for l in [k for k in locs.keys()][:int(len(locs)*0.1)]:
            locs_test.append(l)
        idx=0
        while idx < len(dist):
            if dist[idx] in locs_test:
                test_rssi.append(rssi.pop(idx))
                test_dist.append(dist.pop(idx))
                continue
            idx+=1

        X = np.array([[r] for r in rssi])
        Y = np.array(dist)
        print('---training---', len(X))
        # model = cache('../datasets/model_linear_cache.pkl', LinearRegression().fit, cb_args=((X, Y)))
        # model = BaseRSSIModel()
        # model = cache('../datasets/model_RNDForest_cache.pkl', RandomForestRegressor(max_depth=8, random_state=0).fit, cb_args=((X, Y)))
        #
        # print('---results---')
        # for tx, ty in zip(test_rssi[:100], test_dist[:100]):
        #     print(tx, model.predict([[tx]]), round(ty, 2))
        # y_true, y_pred = test_dist, [model.predict([[tr]]) for tr in test_rssi]
        # print("Mean squared error:", mean_squared_error(y_true, y_pred))

        res_X = []
        errs  = []
        res_Y = []
        for _loc, _rssis in locs.items():
            # standard deviation of rssis in one location
            res_X.append(np.mean(_rssis))
            errs.append(np.std(_rssis))
            res_Y.append(_loc)

        # Visualize resulting model:
        # res_X = np.arange(-20, -90, -1)
        # res_Y = [ model.predict([[x]])[0] for x in res_X ]
        plt.scatter(res_X, res_Y)
        plt.errorbar(res_X, res_Y, xerr=errs, linestyle="None")
    plt.show()



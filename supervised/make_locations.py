""" Transform server stored rssi and sensor data into a data set """
import csv
import os
import matplotlib as mp
import sklearn as sk
import numpy as np


def get_rssi_locations(base="."):
    timestamp= [], rssi= [], sr_x= [], sr_y= [], gw_x= [], gw_y= []
    gw_map = {}
    for sens_dir in os.listdir(base):
        print('reading', sens_dir)
        if os.path.isdir(os.path.join(base, sens_dir)):
            rssi_file = os.listdir(os.path.join(base, sens_dir))[0]
            # Get sensor locations data
            loc_files = os.listdir(os.path.join(base, sens_dir))[1:]
            for sens_loc in loc_files:
                with open(os.path.join(base, sens_dir, sens_loc), 'r') as f:
                    _loc_meta = csv.reader(f)
                    _meta = [x for x in _loc_meta]
                    loc_time_start = int(_meta[1][0])
                    loc_time_stop  = int(_meta[1][1])
                    loc = (_meta[1][2], _meta[1][3])
                    for g in _meta[2:]:
                        if g[0] not in gw_map:
                            gw_map[g[0]] = (g[1], g[2])
                # Get relevant sensor rssis for this location
                sens_time_start = int(rssi_file[9:-4])
                with open(os.path.join(base, sens_dir, rssi_file), "r") as f:
                    _sens_rssi = csv.reader(f)
                    _rssis_meta = [x for x in _sens_rssi]
                    for p in _rssis_meta[2:]:
                        if loc_time_start <= int(p[0])+sens_time_start <= loc_time_stop:
                            timestamp.append((int(p[0]) + sens_time_start))
                            rssi.append(p[2])
                            sr_x.append(loc[0])
                            gw_x.append(gw_map[p[1]][0])
                            sr_y.append(loc[1])
                            gw_y.append(gw_map[p[1]][1])
    return timestamp, rssi, sr_x, sr_y, gw_x, gw_y
    

if __name__ == "__main__":
    timestamp, rssi, sr_x, sr_y, gw_x, gw_y = get_rssi_locations()

""" Transform server stored rssi and sensor data into a python-readable data set
Requires path to a sensor directory of the database data.
Assuming gateways are stationary.
"""
import csv
import os


def get_rssi_locations(base="."):
    timestamp, rssi, sr_x, sr_y, gw_x, gw_y = [], [], [], [], [], []
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
                    loc_time_stop = int(_meta[1][1])
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
                        if loc_time_start <= int(p[0]) + sens_time_start <= loc_time_stop:
                            timestamp.append((int(p[0]) + sens_time_start))
                            rssi.append(int(p[2]))
                            sr_x.append(int(loc[0]))
                            gw_x.append(int(gw_map[p[1]][0]))
                            sr_y.append(int(loc[1]))
                            gw_y.append(int(gw_map[p[1]][1]))
    return timestamp, rssi, sr_x, sr_y, gw_x, gw_y


def get_single_gw_rssi_locations(base=".", gw_n=0):
    timestamp, rssi, sr_x, sr_y, gw_x, gw_y = [], [], [], [], [], []
    gw_x_one, gw_y_one = 0, 0
    gw_map = None
    for sens_dir in os.listdir(base):
        print('reading', sens_dir)
        if os.path.isdir(os.path.join(base, sens_dir)):
            rssi_file = os.listdir(os.path.join(base, sens_dir))[0]
            # Get sensor locations data
            loc_files = os.listdir(os.path.join(base, sens_dir))[1:]
            # Set gateways position
            for sens_loc in loc_files:
                with open(os.path.join(base, sens_dir, sens_loc), 'r') as f:
                    _loc_meta = csv.reader(f)
                    _meta = [x for x in _loc_meta]
                    loc_time_start = int(_meta[1][0])
                    loc_time_stop = int(_meta[1][1])
                    loc = (_meta[1][2], _meta[1][3])
                    for g in _meta[2:]:
                        if not gw_map and str(gw_n)==g[0]:
                            gw_map = g[0]
                            gw_x_one, gw_y_one = int(g[1]), int(g[2])
                # Get relevant sensor rssis for this location
                sens_time_start = int(rssi_file[9:-4])
                with open(os.path.join(base, sens_dir, rssi_file), "r") as f:
                    _sens_rssi = csv.reader(f)
                    _rssis_meta = [x for x in _sens_rssi]
                    # average every 10 seconds into one point
                    tCurrent = 0
                    rssiCurrent = []
                    for p in _rssis_meta[2:]:
                        if loc_time_start <= int(p[0]) + sens_time_start <= loc_time_stop:
                            if gw_map==p[1]:
                                if int(p[0]) - tCurrent > 10000:
                                    if len(rssiCurrent)>0:
                                        timestamp.append((int(tCurrent) + sens_time_start))
                                        rssi.append(int(sum(rssiCurrent)/len(rssiCurrent)))
                                        sr_x.append(int(loc[0]))
                                        gw_x.append(gw_x_one)
                                        sr_y.append(int(loc[1]))
                                        gw_y.append(gw_y_one)
                                    # Update buffer
                                    rssiCurrent = []
                                    tCurrent = int(p[0])
                                rssiCurrent.append(int(p[2]))
    return timestamp, rssi, sr_x, sr_y, gw_x, gw_y
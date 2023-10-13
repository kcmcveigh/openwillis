# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

import json
import logging
import os

import numpy as np
import pandas as pd

from forest.jasmine.data2mobmat import gps_to_mobmat, infer_mobmat
from forest.jasmine.mobmat2traj import imp_to_traj, impute_gps
from forest.jasmine.sogp_gps import bv_select

import util.gps_util as gutil

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()


def create_empty_dataframes(config):

    hourly = pd.DataFrame(
        columns=[
            config["year"],
            config["month"],
            config["day"],
            config["hour"],
            config["observed_time"],
            config["dist_travelled"],
            config["home_time"],
            config["home_max_dist"],
            config["home_mean_dist"],
        ]
    )
    daily = pd.DataFrame(
        columns=[
            config["year"],
            config["month"],
            config["day"],
            config["observed_time"],
            config["observed_time_day"],
            config["observed_time_night"],
            config["dist_travelled"],
            config["home_time"],
            config["home_max_dist"],
            config["home_mean_dist"],
        ]
    )
    summary = pd.DataFrame(
        columns=[
            config["no_days"],
            config["total_observed_time"],
            config["mean_dist_travelled"],
            config["sd_dist_travelled"],
            config["mean_home_time"],
            config["sd_home_time"],
            config["mean_home_max_dist"],
            config["sd_home_max_dist"],
            config["mean_home_mean_dist"],
            config["sd_home_mean_dist"],
        ]
    )

    return [hourly, daily, summary]


def get_config(filepath, json_file):
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    filepath : str
        The path to the configuration file.
    json_file : str
        The name of the configuration file.

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(filepath)
    measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))

    file = open(measure_path)
    measures = json.load(file)
    return measures


def gps_analysis(filepath, timezone):

    measures = get_config(os.path.abspath(__file__), "gps.json")
    df_list = create_empty_dataframes(measures)

    try:
        # read data from csv
        data = pd.read_csv(filepath)

        if data.shape == (0, 0):
            raise ValueError("No data available.")

        # quality check
        gutil.gps_quality(data)
        mean_acc = np.mean(data.accuracy)

        mobmat1 = gps_to_mobmat(data, 10, 51, 51, mean_acc, 51)
        mobmat2 = infer_mobmat(mobmat1, 10, 51)
        out_dict = bv_select(
            mobmat2,
            0.01,
            0.05,
            100,
            [60 * 60 * 24 * 10, 60 * 60 * 24 * 30, 0.002, 5, 1, 0.3, 0.2, 0.5],
            None,
            None,
        )
        imp_table = impute_gps(
            mobmat2,
            bv_set,
            "GLC",
            3,
            10,
            2,
            timezone,
            [60 * 60 * 24 * 10, 60 * 60 * 24 * 30, 5, 1, 0.3, 0.2, 0.5, 200],
        )
        traj = imp_to_traj(imp_table, mobmat2, mean_acc)
        # raise error if traj coordinates are not in the range of
        # [-90, 90] and [-180, 180]
        if traj.shape[0] > 0:
            if (
                np.max(traj[:, 1]) > 90
                or np.min(traj[:, 1]) < -90
                or np.max(traj[:, 2]) > 180
                or np.min(traj[:, 2]) < -180
                or np.max(traj[:, 4]) > 90
                or np.min(traj[:, 4]) < -90
                or np.max(traj[:, 5]) > 180
                or np.min(traj[:, 5]) < -180
            ):
                raise ValueError(
                    "Trajectory coordinates are not in the range of "
                    "[-90, 90] and [-180, 180]."
                )

        df_list = gutil.trajectory_statistics(traj, df_list, timezone)

    except Exception as e:
        logger.error(f"Error in gps summary calculation- file: {filepath} & Error: {e}")

    finally:
        hourly, daily, summary = df_list

        if hourly.shape[0] == 0:
            hourly.loc[0] = pd.NA
        if daily.shape[0] == 0:
            daily.loc[0] = pd.NA
        if summary.shape[0] == 0:
            summary.loc[0] = pd.NA

    return hourly, daily, summary

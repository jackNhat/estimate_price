from __future__ import print_function
import argparse
import warnings

from helpers import analysis, final_data

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="path to csv data file")
ap.add_argument("-a", "--analysis", type=bool, default=False,
                help="It will run analysis function")
ap.add_argument("-f", "--final_data", type=bool, default=False, help="it will run final_data function")
args = vars(ap.parse_args())

path = args["path"]


if __name__ == '__main__':

    if args["analysis"]:
        analysis(path)
    elif args["final_data"]:
        final_data(path)
    else:
        print("No function executed")

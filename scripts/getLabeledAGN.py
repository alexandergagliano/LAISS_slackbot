from source import *
from astropy.time import Time
import tempfile
import os
import astro_ghost
from astro_ghost.ghostHelperFunctions import getTransientHosts,getGHOST
from astro_ghost.NEDQueryFunctions import getNEDInfo
from astro_ghost.photoz_helper import calc_photoz
import requests as req
from auth import toku
import glob
from build_rec_test import post as pst
from build_rec_test import build_rec as bs
import time
import argparse
from datetime import datetime, timedelta
import subprocess
from astropy.coordinates import SkyCoord
from astropy import units as u
from shapley import dict_merge, plot_shap, upload_file, make_file_public, share_public_link, shap_values

import antares_client
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Run LAISS AD. Ex: python3 auto.py 2 D05R7RK4K8T')
# Add command-line arguments for input, new data, and output file paths
parser.add_argument('lookback_t', help='lookback_t')
args = parser.parse_args()

# Get current date
current_date = datetime.now()
year = current_date.year
month = current_date.month
day = current_date.day

# Calculate today's MJD
today_mjd = calculate_mjd(year, month, day)
print("Today's Modified Julian Date:", today_mjd)

lookback_t = float(args.lookback_t)
print(f"Looking back to all objects with alerts between MJD {today_mjd}-{today_mjd - lookback_t}:")

# Get list of tagged loci
LAISS_RFC_AD_loci = antares_client.search.search(
    {
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "tags": [
                                "LAISS_RFC_AD_filter"
                            ]
                        }
                    }
                ],
                "must": {
                    "range": {
                        "properties.newest_alert_observation_time": {
                            "gte": today_mjd - lookback_t
                        }
                    }
                }
            }
        }
    }
)

# https://stackoverflow.com/questions/1528237/how-to-handle-exceptions-in-a-list-comprehensions
def catch(func, *args, handle=lambda e : e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)

LAISS_RFC_AD_locus_ids = [catch(lambda : l.locus_id) for l in LAISS_RFC_AD_loci] # to catch requests.exceptions.JSONDecodeError: [Errno Expecting ',' delimiter]
print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")

AGN_names = []
AGN_labels= []

for l in LAISS_RFC_AD_locus_ids:
    if l.startswith("ANT"):  # take any objects 
        print(l)
        locus = antares_client.search.get_by_id(l)
        AGN_names.append(l)
        if ('tns_public_objects' not in locus.catalogs):
            if ('LAISS_RFC_anomaly_score' in locus.properties) and (('gaia_dr3_variability' in locus.catalogs) or ('veron_agn_qso' in locus.catalogs) or ('vsx' in locus.catalogs)):
                AGN_labels.append(1)
            else:
                AGN_labels.append(0)
 
        else:
            if ('LAISS_RFC_anomaly_score' in locus.properties) and (('gaia_dr3_variability' in locus.catalogs) or ('veron_agn_qso' in locus.catalogs) or ('vsx' in locus.catalogs)):
                AGN_labels.append(1)
            else:
                AGN_labels.append(0)

pd.DataFrame({'Transient':np.array(AGN_names), 'Catalog AGN?':np.array(AGN_labels)}).to_csv("PassiveLabelingResults_AGN2024.csv", index=False)

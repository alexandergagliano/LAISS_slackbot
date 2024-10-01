from source import *
from astropy.time import Time
import tempfile
import os
from astropy.io import fits, ascii
import glob
from build_rec import post as pst
from build_rec import build_rec as bs
import time
import argparse
from datetime import datetime
import astro_ghost
from astro_ghost.PS1QueryFunctions import ps1cone
import antares_client
import datetime
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Run LAISS AD. Ex: python3 get_AD_hosts.py 30 10')
# Add command-line arguments for input, new data, and output file paths
parser.add_argument('lookback_t', help='lookback_t')
parser.add_argument('anom_thresh', help='anom_thresh')
args = parser.parse_args()

# Get current date
current_date = datetime.datetime.now()
year = current_date.year
month = current_date.month
day = current_date.day

# Calculate today's MJD
today_mjd = calculate_mjd(year, month, day)
print("Today's Modified Julian Date:", today_mjd)

lookback_t = float(args.lookback_t)
print(f"Looking back to all objects tagged within MJD {today_mjd}-{today_mjd - lookback_t}:")

anom_thresh = float(args.anom_thresh) # anomaly threshold
print(f"Using Anomaly Threshold of: {anom_thresh}%")

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
LAISS_RFC_AD_locus_ids = np.unique(LAISS_RFC_AD_locus_ids)

print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")

GHOSTpath = os.getenv('GHOST_PATH')
if not GHOSTpath:
    try:
        GHOSTpath = astro_ghost.__file__
        GHOSTpath = GHOSTpath.split("/")[:-1]
        GHOSTpath = "/".join(GHOSTpath)
    except:
        print("Error! I don't know where you installed GHOST -- set GHOST_PATH as an environmental variable or pass in the GHOSTpath parameter.")

fullTable = pd.read_csv(GHOSTpath+"/database/GHOST.csv")

#ignore the ones already in the table
LAISS_RFC_AD_locus_ids = np.array(list(set(LAISS_RFC_AD_locus_ids) - set(fullTable['TransientName'].values)))

host_DF = []

for l in LAISS_RFC_AD_locus_ids:
    if l.startswith("ANT"):  # only take objects from this year
        locus = antares_client.search.get_by_id(l)
        hostRA = locus.properties['raMean']
        hostDec = locus.properties['decMean']
        a = ps1cone(hostRA, hostDec, 10./3600)
         
        if a:
            a = ascii.read(a)
            a = a.to_pandas()
            ps1match = a.iloc[[0]]

            # TNS objects below:
            try:
                tns = locus.catalog_objects['tns_public_objects'][0]
                best_name, best_cls = tns['name'], tns['type']
            except:
                best_name = l
                best_cls = ''

            ps1match['TransientName'] = best_name
            ps1match['TransientClass'] = best_cls
            ps1match['TransientRA'] = locus.ra
            ps1match['TransientDEC'] = locus.dec

            host_DF.append(ps1match)

#add to GHOST catalog
host_DF = pd.concat(host_DF, ignore_index=True)
fullTable = pd.concat([fullTable, host_DF], ignore_index=True).drop_duplicates(subset=['TransientName'])
fullTable.to_csv(GHOSTpath+"/database/GHOST.csv",index=False)
print(f"Done storing {len(host_DF)} hosts.\n")

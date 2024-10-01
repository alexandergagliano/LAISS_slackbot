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

working_dir = '/root/LAISS/AD_slackbot/'

data_dir = os.path.join(working_dir, 'data')
log_dir = os.path.join(working_dir, 'logs')

parser = argparse.ArgumentParser(description='Run LAISS AD. Ex: python3 auto.py 2 D05R7RK4K8T')
# Add command-line arguments for input, new data, and output file paths
parser.add_argument('lookback_t', help='lookback_t')
parser.add_argument('anom_thresh', help='anom_thresh')
parser.add_argument('channel', help='slack channel ID')
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
channel = str(args.channel)
print(f"Looking back to all objects with alerts between MJD {today_mjd}-{today_mjd - lookback_t}:")

anom_thresh = float(args.anom_thresh) # classification threshold

print(f"Using Classification Threshold of: {anom_thresh}%")

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

def getLastPosted(numDays=5):
    print(f"Excluding events already posted in the last {numDays} days.")
    p1=req.get('https://slack.com/api/conversations.history',
	params={'channel':channel, 'parse':'none'},
	 headers={'Authorization': f'Bearer {toku}'})

    p1.raise_for_status()

    # Additional check for Slack-specific errors in the response body
    response_data = p1.json()
    if not response_data.get('ok'):
        raise ValueError(f"Failed to check posting history: {response_data.get('error')}")

    posted_names = []
    posted_ts = []

    for message in response_data['messages']:
        try:
            names = [x['title'].split(" ")[1] for x in message['attachments']] 
            ts = [message['ts']]*len(names)
            posted_names.append(names)
            posted_ts.append(ts)
        except:
            continue
    df = pd.DataFrame({'Transient':np.concatenate(posted_names), 'TimeStamp':np.concatenate(posted_ts)})

    # Get the current datetime
    current_datetime = datetime.now()

    # Subtract N days
    num_days_ago = current_datetime - timedelta(days=numDays)

    timestamp_num_days_ago = num_days_ago.timestamp()
   
    df = df[df['TimeStamp'].astype("float") > timestamp_num_days_ago]

    #the names of the transients posted <numDays ago
    #include the events in the locally-stored db 
    df_posted = pd.read_csv(os.path.join(data_dir, "anomalies_db.csv"))
    df_posted.rename(columns={"ANTID":'Transient'}, inplace=True)
    df_comb = pd.concat([df_posted, df], ignore_index=True)
    print("Number of total posted transients:")
    print(len(np.unique(df_comb['Transient'])))
    return np.unique(df_comb['Transient'].values)

LAISS_RFC_AD_locus_ids = [catch(lambda : l.locus_id) for l in LAISS_RFC_AD_loci] # to catch requests.exceptions.JSONDecodeError: [Errno Expecting ',' delimiter]
print(f"Considering {len(LAISS_RFC_AD_locus_ids)} candidates...")

g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_anom_score_l, g50_ra_l, g50_dec_l = [], [], [], [], [], []
#get brightest mag, and brightest phase
g50_brightmag_l, g50_brightphase_l = [], []
g50_hostname_l, g50_hostsep_l, g50_host_photoz_l = [], [], []
g50_ra_l, g50_dec_l = [], []
g50_host_redshift_flag_l = []
g50_firstphase_l = []
g50_firstmag_l = []
g50_ztfname_l = []

#exclude ones that have already been posted in the last many days (...at all)
postedNames = getLastPosted(numDays=100000)

shap_features = {}

with open(os.path.join(data_dir, "host_features.txt")) as host_f:
    host_features = [line.strip() for line in host_f.readlines()]

with open(os.path.join(data_dir, "lc_features.txt")) as lc_f:
    lc_features = [line.strip() for line in lc_f.readlines()]

lc_and_host_features = lc_features + host_features


for l in LAISS_RFC_AD_locus_ids:
    if l.startswith("ANT"):  # take any objects
        #if l in postedNames:
        #    continue
        locus = antares_client.search.get_by_id(l)
        duration = locus.properties['newest_alert_observation_time'] - locus.properties['oldest_alert_observation_time']
        #if duration < maxDuration: 
        #    continue
        if ('tns_public_objects' not in locus.catalogs):
            if ('LAISS_RFC_anomaly_score' in locus.properties) and (locus.properties['LAISS_RFC_anomaly_score'] >= anom_thresh) and ('gaia_dr3_variability' not in locus.catalogs) and ('veron_agn_qso' not in locus.catalogs) and ('bright_guide_star_cat' not in locus.catalogs) and ('vsx' not in locus.catalogs):
 
                #first, store the shapley values 
                shap_features[l] = {}
                for feat in lc_and_host_features:
                    try:
                        shap_features[l][feat] = locus.properties[feat]
                    except:
                        shap_features[l][feat] = np.nan

                #then, store everything else
                g50_antid_l.append(l), g50_tns_name_l.append("No TNS"), g50_tns_cls_l.append("---"),
                g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)
                g50_brightmag_l.append(locus.properties['brightest_alert_magnitude'])
                g50_brightphase_l.append(today_mjd - locus.properties['brightest_alert_observation_time'])
                g50_firstphase_l.append(today_mjd - locus.properties['oldest_alert_observation_time'])
                g50_firstmag_l.append(locus.properties['oldest_alert_magnitude'])
                g50_ztfname_l.append(locus.properties['ztf_object_id'])

                #get host properties
                tns_cls = ''
                temp_redshift_flag = 'PHOT'

                hosts = pd.DataFrame({'objID':[locus.properties['objID']], 'raMean':[locus.properties['raMean']], 'decMean':[locus.properties['decMean']],'TransientRA':[locus.ra], 'TransientDEC':[locus.dec], 'TransientName':[l]})
                hosts = getNEDInfo(hosts)

                # Coordinates of the first object (RA, Dec)
                coord1 = SkyCoord(ra=locus.ra*u.degree, dec=locus.dec*u.degree, frame='icrs')
                coord2 = SkyCoord(ra=locus.properties['raMean']*u.degree, dec=locus.properties['decMean']*u.degree, frame='icrs')
                hosts['dist'] = coord1.separation(coord2).arcsec


                if len(hosts) > 0:
                    if (hosts['NED_name'].values[0] == hosts['NED_name'].values[0]) and (len(hosts['NED_name'].values[0]) > 0):
                        g50_hostname_l.append(hosts['NED_name'].values[0])
                    else:
                        g50_hostname_l.append('ObjID %i'%hosts['objID'].values[0])
                    g50_hostsep_l.append(hosts['dist'].values[0])

                    # get NED redshift if available
                    if (hosts['NED_redshift'].values[0] != hosts['NED_redshift'].values[0]):
                        try:
                            hosts = calc_photoz(hosts)
                        except:
                            hosts['photo_z'] = -9
                    else:
                            hosts['photo_z'] = hosts['NED_redshift'].values[0]
                            if hosts['NED_redshift_flag'].values[0] == 'SPEC':
                                temp_redshift_flag = 'SPEC'
                    g50_host_photoz_l.append(hosts['photo_z'].values[0])
                else:
                    g50_hostname_l.append('---')
                    g50_hostsep_l.append(-9)
                    g50_host_photoz_l.append(-9)
                g50_host_redshift_flag_l.append(temp_redshift_flag)

        else:
            # TNS objects below:
            try:
                tns = locus.catalog_objects['tns_public_objects'][0]
                tns_name, tns_cls, tns_redshift = tns['name'], tns['type'], tns['redshift']
                best_name = tns_name
            except:
                print(f"{l} likely is on TNS but is outside of 1 arcsec matching for catalogs...Check!")
                tns_cls = ''  
                tns_redshift = -9
                tns_name = '---'

                best_name = l

            if tns_cls == '': tns_cls = "---"

            if ('LAISS_RFC_anomaly_score' in locus.properties) and (locus.properties['LAISS_RFC_anomaly_score'] >= anom_thresh) and ('gaia_dr3_variability' not in locus.catalogs) and ('veron_agn_qso' not in locus.catalogs) and ('bright_guide_star_cat' not in locus.catalogs) and ('vsx' not in locus.catalogs) and (tns_cls == "---"):

                #first, store the shapley values
                shap_features[l] = {}
                for feat in lc_and_host_features:
                    try:
                        shap_features[l][feat] = locus.properties[feat]
                    except:
                        shap_features[l][feat] = np.nan

                g50_antid_l.append(l), g50_tns_name_l.append(tns_name), g50_tns_cls_l.append(tns_cls),
                g50_anom_score_l.append(locus.properties['LAISS_RFC_anomaly_score'])
                g50_ra_l.append(locus.ra), g50_dec_l.append(locus.dec)
                g50_brightmag_l.append(locus.properties['brightest_alert_magnitude'])
                g50_brightphase_l.append(today_mjd - locus.properties['brightest_alert_observation_time'])
                g50_firstphase_l.append(today_mjd - locus.properties['oldest_alert_observation_time'])
                g50_firstmag_l.append(locus.properties['oldest_alert_magnitude'])
                g50_ztfname_l.append(locus.properties['ztf_object_id'])

                temp_redshift_flag = 'PHOT'

                hosts = pd.DataFrame({'objID':[locus.properties['objID']], 'raMean':[locus.properties['raMean']], 'decMean':[locus.properties['decMean']],'TransientRA':[locus.ra], 'TransientDEC':[locus.dec], 'TransientName':[l]})
                hosts = getNEDInfo(hosts)

                # Coordinates of the first object (RA, Dec)
                coord1 = SkyCoord(ra=locus.ra*u.degree, dec=locus.dec*u.degree, frame='icrs')
                coord2 = SkyCoord(ra=locus.properties['raMean']*u.degree, dec=locus.properties['decMean']*u.degree, frame='icrs')
                hosts['dist'] = coord1.separation(coord2).arcsec

                if len(hosts) > 0:
                    if (hosts['NED_name'].values[0] == hosts['NED_name'].values[0]) and (len(hosts['NED_name'].values[0]) > 0):
                        g50_hostname_l.append(hosts['NED_name'].values[0])
                    else:
                        g50_hostname_l.append('ObjID %i'%hosts['objID'].values[0])
                    g50_hostsep_l.append(hosts['dist'].values[0])

                    # get NED redshift if available
                    if (hosts['NED_redshift'].values[0] != hosts['NED_redshift'].values[0]) and (tns_redshift == None):
                        try:
                            hosts = calc_photoz(hosts)
                        except:
                            hosts['photo_z'] = -9
                    else:
                            hosts['photo_z'] = hosts['NED_redshift'].values[0]
                            if hosts['NED_redshift_flag'].values[0] == 'SPEC':
                                temp_redshift_flag = 'SPEC'
                    final_z = hosts['photo_z'].values[0]
                else:
                    g50_hostname_l.append('---')
                    g50_hostsep_l.append(-9)
                    final_z = -9
                if tns_redshift != None:
                    final_z = tns_redshift
                    temp_redshift_flag = 'SPEC'
                g50_host_photoz_l.append(final_z)
                g50_host_redshift_flag_l.append(temp_redshift_flag)

final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l, final_cand_brightmag_l, final_cand_brightphase_l, final_cand_firstmag_l, final_cand_firstphase_l, final_cand_hostname_l, final_cand_hostsep_l, final_cand_host_photoz_l, final_cand_ra_l, final_cand_dec_l, final_cand_host_redshift_flag_l, final_cand_ztfname_l  = g50_antid_l, g50_tns_name_l, g50_tns_cls_l, g50_anom_score_l, g50_brightmag_l, g50_brightphase_l, g50_firstmag_l, g50_firstphase_l, g50_hostname_l, g50_hostsep_l, g50_host_photoz_l, g50_ra_l, g50_dec_l, g50_host_redshift_flag_l, g50_ztfname_l

def run(post=True):
    global final_cand_antid_l, final_cand_hostsep_l, final_cand_anom_score_l

    ps= []
    att= []
    bs_set = []
    posted_names = []
    if post:
        #ps.append(f"Duration-selected (>{maxDuration}d) anomalous candidates for today:\n")
        ps.append("Anomalous candidates for today:\n")
        Ntoomany =  5
        offsetCut = False 

        idxs = np.arange(len(final_cand_antid_l))
        #turn off offset cut for now

        final_cand_hostsep_l = np.array(final_cand_hostsep_l)
        final_cand_anom_score_l = np.array(final_cand_anom_score_l)

        #subset by those >1'' from the host galaxy center
        #final_cand_hostsep_l = np.array(final_cand_hostsep_l)
        #offsetBool = final_cand_hostsep_l <= 1

        #filtered_anom_scores = final_cand_anom_score_l[offsetBool]
        #indices_filtered = np.argsort(-filtered_anom_scores)
        #original_indices = np.arange(len(final_cand_anom_score_l))[offsetBool]

        # Map top filtered indices back to original indices
        #idxs = original_indices[indices_filtered]

        #if len(idxs) > 0:
        #    ps[0] += f"Limiting to those <1\'\' from a host...\n"
        #else:
        #if len(idxs) < 1:
        #    #go back to posting all candidates
        #    idxs = np.arange(len(final_cand_antid_l))

        if (len(final_cand_antid_l) > Ntoomany) | (offsetCut):
            if (len(final_cand_antid_l) > Ntoomany):
                ps[0] += f"Greater than {Ntoomany} anomalous transients, limiting to at most 5 with highest anomaly scores...\n"

                final_cand_anom_score_l = np.array(final_cand_anom_score_l)
            
                idxs = np.argsort(-final_cand_anom_score_l)[:5]

            for idx in idxs:
                print(f"Has final_cand_antid_l[idx] been posted before?")
                print(final_cand_antid_l[idx] in postedNames)
                try:
                    shap_url = shap_values(final_cand_antid_l[idx], shap_features[final_cand_antid_l[idx]], 'C078CJZE3K5')
                    ant = bs(antaresID=final_cand_antid_l[idx], tns_name=final_cand_tns_name_l[idx], tns_cls=final_cand_tns_cls_l[idx], anom_score=final_cand_anom_score_l[idx], brightest_mag=final_cand_brightmag_l[idx], brightest_phase=final_cand_brightphase_l[idx], first_mag=final_cand_firstmag_l[idx], first_phase=final_cand_firstphase_l[idx], host_name=final_cand_hostname_l[idx], host_sep=final_cand_hostsep_l[idx], host_photoz=final_cand_host_photoz_l[idx], host_z_flag=final_cand_host_redshift_flag_l[idx], ra=final_cand_ra_l[idx], dec=final_cand_dec_l[idx], ztfname=final_cand_ztfname_l[idx], shap_url=shap_url, posted_before=(final_cand_antid_l[idx] in postedNames))
                except:
                    print(f"Some error while trying to post for: {antaresID} {tns_name} {tns_cls} {anom_score}. Skip!")
                    continue
                #don't post duplicates
                if final_cand_antid_l[idx] not in posted_names:
                    bs_set.append(ant)
                    posted_names.append(final_cand_antid_l[idx])
        else:
            for antaresID, tns_name, tns_cls, anom_score, brightmag, brightphase, first_mag, firstphase, hostname, hostsep, host_z, host_z_flag, ra, dec, ztfname in zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l, final_cand_anom_score_l, final_cand_brightmag_l, final_cand_brightphase_l, final_cand_firstmag_l, final_cand_firstphase_l, final_cand_hostname_l, final_cand_hostsep_l, final_cand_host_photoz_l, final_cand_host_redshift_flag_l, final_cand_ra_l, final_cand_dec_l, final_cand_ztfname_l):
                #try: 
                if True:
                    shap_url = shap_values(antaresID, shap_features[antaresID], 'C078CJZE3K5')

                    ant = bs(antaresID=antaresID, tns_name=tns_name, tns_cls=tns_cls, anom_score=anom_score, brightest_mag=brightmag, brightest_phase=brightphase, first_mag=first_mag, first_phase=firstphase, host_name=hostname, host_sep=hostsep, host_photoz=host_z, host_z_flag=host_z_flag, ra=ra, dec=dec, ztfname=ztfname, shap_url=shap_url, posted_before=(antaresID in postedNames))
                #except:
                #    print(f"Some error while trying to post for: {antaresID} {tns_name} {tns_cls} {anom_score}. Skip!")
                #    continue
                #don't post duplicates
                if antaresID not in posted_names:
                    bs_set.append(ant)
                    posted_names.append(antaresID)
            if len(final_cand_antid_l) < 1:
                ps = []
        pst(bs_set,ps,lookback_t,today_mjd,channel=channel)

    return 0

def save_objects(file):
    if not os.path.exists(file):
        df = pd.DataFrame(zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l),
                          columns=['ANTID', 'TNS_Name', 'Spec_Class'])
        df.to_csv(file)
    else:
        df = pd.read_csv(file)
        df2 = pd.DataFrame(zip(final_cand_antid_l, final_cand_tns_name_l, final_cand_tns_cls_l),
                          columns=['ANTID', 'TNS_Name', 'Spec_Class'])
        df.set_index('ANTID', inplace=True)
        df2.set_index('ANTID', inplace=True)
        # Concatenate the two dataframes along the rows
        merged_df = pd.concat([df, df2])
        # keep the last occurrence of each duplicate row -- (most up to date)
        merged_df = merged_df.drop_duplicates(keep='last')
        merged_df.to_csv(file) # overwrite the old file with unique new + old objects


def save_log():
    # At the end of your script, write the current date to a logfile
    with open(os.path.join(log_dir, 'logfile.txt'), 'a') as logfile:
        logfile.write(f'{datetime.now()}\n')

if __name__ == '__main__':
    run()
    save_objects(file=os.path.join(data_dir, "anomalies_db.csv"))

#def main():
#    run()
#    save_objects(file='/root/LAISS/AD_slackbot/data/anomalies_db.csv')
#    save_log()
#    return 0

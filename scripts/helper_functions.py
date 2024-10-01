import astro_ghost
from astro_ghost.ghostHelperFunctions import getGHOST
import light_curve as lc
import numpy as np
import math
import pandas as pd
from pathlib import Path
import time
import antares_client
from astropy.table import MaskedColumn
from itertools import chain
from astropy.coordinates import SkyCoord
from PIL import Image
from astropy.io import fits
import astropy.units as u
import os
import tempfile
import matplotlib.pyplot as plt

def get_base_name(path):
    p = Path(path)
    while True:
        stem = p.stem
        if stem == p.name:  # No more extensions to strip
            break
        p = Path(stem)  # Prepare for next iteration, if needed
    return stem

# GHOST getTransientHosts function with timeout
from timeout_decorator import timeout, TimeoutError
@timeout(120)  # Set a timeout of 60 seconds to query GHOST throughput APIs for host galaxy data
def getTransientHosts_with_timeout(**args):
    return astro_ghost.ghostHelperFunctions.getTransientHosts(**args)


# Functions to extract light-curve features
def replace_magn_with_flux(s):
    if 'magnitude' in s:
        return s.replace('magnitudes', 'fluxes').replace('magnitude', 'flux')
    return f'{s} for flux light curve'

def create_base_features_class(
        magn_extractor,
        flux_extractor,
        bands=('R', 'g',),
    ):
    feature_names = ([f'{name}_magn' for name in magn_extractor.names]
                     + [f'{name}_flux' for name in flux_extractor.names])

    property_names = {band: [f'feature_{name}_{band}'.lower()
                             for name in feature_names]
                      for band in bands}

    features_count = len(feature_names)

    return feature_names, property_names, features_count

###### calculate relevant light curve features ########
MAGN_EXTRACTOR = lc.Extractor(
    lc.Amplitude(),
    lc.AndersonDarlingNormal(),
    lc.BeyondNStd(1.0),
    lc.BeyondNStd(2.0),
    lc.Cusum(),
    lc.EtaE(),
    lc.InterPercentileRange(0.02),
    lc.InterPercentileRange(0.1),
    lc.InterPercentileRange(0.25),
    lc.Kurtosis(),
    lc.LinearFit(),
    lc.LinearTrend(),
    lc.MagnitudePercentageRatio(0.4, 0.05),
    lc.MagnitudePercentageRatio(0.2, 0.05),
    lc.MaximumSlope(),
    lc.Mean(),
    lc.MedianAbsoluteDeviation(),
    lc.PercentAmplitude(),
    lc.PercentDifferenceMagnitudePercentile(0.05),
    lc.PercentDifferenceMagnitudePercentile(0.1),
    lc.MedianBufferRangePercentage(0.1),
    lc.MedianBufferRangePercentage(0.2),
    lc.Periodogram(
        peaks=5,
        resolution=10.0,
        max_freq_factor=2.0,
        nyquist='average',
        fast=True,
        features=(
            lc.Amplitude(),
            lc.BeyondNStd(2.0),
            lc.BeyondNStd(3.0),
            lc.StandardDeviation(),
        ),
    ),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StandardDeviation(),
    lc.StetsonK(),
    lc.WeightedMean(),
)

FLUX_EXTRACTOR = lc.Extractor(
    lc.AndersonDarlingNormal(),
    lc.Cusum(),
    lc.EtaE(),
    lc.ExcessVariance(),
    lc.Kurtosis(),
    lc.MeanVariance(),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StetsonK(),
)

def remove_simultaneous_alerts(table):
    """Remove alert duplicates"""
    dt = np.diff(table['ant_mjd'], append=np.inf)
    return table[dt != 0]

def get_detections(photometry, band):
    """Extract clean light curve in given band from locus photometry"""
    band_lc = photometry[(photometry['ant_passband'] == band) & (~photometry['ant_mag'].isna())]
    idx = ~MaskedColumn(band_lc['ant_mag']).mask
    detections = remove_simultaneous_alerts(band_lc[idx])
    return detections



def plot_RFC_prob_vs_lc_ztfid(clf, anom_ztfid, anom_spec_cls, anom_spec_z, anom_thresh, lc_and_hosts_df, lc_and_hosts_df_120d, ref_info, savefig, figure_path):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    #try:
    pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
    pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
    pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
    num_anom_epochs = len(np.where(pred_prob_anom[:, 1]>=anom_thresh)[0])
#    except:
    #    print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
    #    return

    try:
        anom_idx = lc_and_hosts_df.iloc[np.where(pred_prob_anom[:, 1]>=anom_thresh)[0][0]].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}.")
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    ref_info = ref_info

    df_ref = ref_info.timeseries.to_pandas()

    df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

    mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
    mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,10))
    ax1.invert_yaxis()
    ax1.errorbar(x=df_ref_r.ant_mjd, y=df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r', label=r'ZTF-$r$')
    ax1.errorbar(x=df_ref_g.ant_mjd, y=df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g', label=r'ZTF-$g$')
    if anom_idx_is == True:
        ax1.axvline(x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
                    label="Tag anomalous", color='dodgerblue', ls='--')
        #ax1.axvline(x=59323, label="Orig. spectrum", color='darkviolet', ls='-.')
        mjd_cross_thresh = round(lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3)

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left)/(right - left)
        #mjd_anom_per2 = (59323 - left)/(right - left)
        plt.text(mjd_anom_per+0.073, -0.075, f"t$_a$ = {int(mjd_cross_thresh)}", horizontalalignment='center',
         verticalalignment='center', transform=ax1.transAxes, fontsize=16, color='dodgerblue')
        #plt.text(mjd_anom_per2+0.12, 0.035, f"t$_s$ = {int(59323)}", horizontalalignment='center',
        # verticalalignment='center', transform=ax1.transAxes, fontsize=16, color='darkviolet')
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f'https://alerce.online/object/{anom_ztfid}')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 0], drawstyle='steps', label=r'$p(Normal)$')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 1], drawstyle='steps', label=r'$p(Anomaly)$')

    if anom_spec_z is None:
            anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(fr"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z})" , pad=25)
    plt.xlabel('MJD')
    ax1.set_ylabel('Magnitude')
    ax2.set_ylabel('Probability (%)')

    if anom_idx_is == True: ax1.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.0,1.12), frameon=False, fontsize=14)
    #if anom_idx_is == True: ax1.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.05,1.12), columnspacing=0.65, frameon=False, fontsize=14)
    else: ax1.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.75,1.12), frameon=False, fontsize=14)
    ax2.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.87,1.12), frameon=False, fontsize=14)

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf", dpi=300, bbox_inches='tight')

    plt.show()

def plot_RFC_prob_vs_lc_yse_IAUid(clf, IAU_name, anom_ztfid, anom_spec_cls, anom_spec_z, anom_thresh, lc_and_hosts_df, lc_and_hosts_df_120d, yse_lightcurve, savefig, figure_path):
    anom_thresh = anom_thresh
    anom_obj_df = lc_and_hosts_df_120d

    try:
        pred_prob_anom = 100 * clf.predict_proba(anom_obj_df)
        pred_prob_anom[:, 0] = [round(a, 1) for a in pred_prob_anom[:, 0]]
        pred_prob_anom[:, 1] = [round(b, 1) for b in pred_prob_anom[:, 1]]
        num_anom_epochs = len(np.where(pred_prob_anom[:, 1]>=anom_thresh)[0])
    except:
        print(f"{anom_ztfid} has some NaN host galaxy values from PS1 catalog. Skip!")
        return

    try:
        anom_idx = lc_and_hosts_df.iloc[np.where(pred_prob_anom[:, 1]>=anom_thresh)[0][0]].obs_num
        anom_idx_is = True
        print("Anomalous during timeseries!")

    except:
        print(f"Prediction doesn't exceed anom_threshold of {anom_thresh}% for {anom_ztfid}.")
        anom_idx_is = False

    max_anom_score = max(pred_prob_anom[:, 1])
    print("max_anom_score", round(max_anom_score, 1))
    print("num_anom_epochs", num_anom_epochs)

    ztf_id_ref = anom_ztfid

    df_ref_g = yse_lightcurve[(yse_lightcurve.FLT == 'g')]
    df_ref_r = yse_lightcurve[(yse_lightcurve.FLT == 'R')]

    mjd_idx_at_min_mag_r_ref = df_ref_r[['MAG']].reset_index().idxmin().MAG
    mjd_idx_at_min_mag_g_ref = df_ref_g[['MAG']].reset_index().idxmin().MAG

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7,10))
    ax1.invert_yaxis()

    df_ref_g_ztf = df_ref_g[df_ref_g.TELESCOPE == 'P48']
    df_ref_g_ps1 = df_ref_g[df_ref_g.TELESCOPE == 'Pan-STARRS1']
    df_ref_r_ztf = df_ref_r[df_ref_r.TELESCOPE == 'P48']
    df_ref_r_ps1 = df_ref_r[df_ref_r.TELESCOPE == 'Pan-STARRS1']

    ax1.errorbar(x=df_ref_r_ztf.MJD, y=df_ref_r_ztf.MAG, yerr=df_ref_r_ztf.MAGERR, fmt='o', c='r', label=r'ZTF-$r$')
    ax1.errorbar(x=df_ref_g_ztf.MJD, y=df_ref_g_ztf.MAG, yerr=df_ref_g_ztf.MAGERR, fmt='o', c='g', label=r'ZTF-$g$')
    ax1.errorbar(x=df_ref_r_ps1.MJD, y=df_ref_r_ps1.MAG, yerr=df_ref_r_ps1.MAGERR, fmt='s', c='r', label=r'PS1-$r$')
    ax1.errorbar(x=df_ref_g_ps1.MJD, y=df_ref_g_ps1.MAG, yerr=df_ref_g_ps1.MAGERR, fmt='s', c='g', label=r'PS1-$g$')

    if anom_idx_is == True:
        ax1.axvline(x=lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0],
                    label="Tagged anomalous", color='darkblue', ls='--')
        mjd_cross_thresh = round(lc_and_hosts_df[lc_and_hosts_df.obs_num == anom_idx].mjd_cutoff.values[0], 3)

        left, right = ax1.get_xlim()
        mjd_anom_per = (mjd_cross_thresh - left)/(right - left)
        plt.text(mjd_anom_per+0.073, -0.075, f"t = {int(mjd_cross_thresh)}", horizontalalignment='center',
         verticalalignment='center', transform=ax1.transAxes, fontsize=16)
        print("MJD crossed thresh:", mjd_cross_thresh)

    print(f'https://ziggy.ucolick.org/yse/transient_detail/{IAU_name}/')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 0], label=r'$p(Normal)$')
    ax2.plot(lc_and_hosts_df.mjd_cutoff, pred_prob_anom[:, 1], label=r'$p(Anomaly)$')

    if anom_spec_z is None:
            anom_spec_z = "None"
    elif isinstance(anom_spec_z, float):
        anom_spec_z = round(anom_spec_z, 3)
    else:
        anom_spec_z = anom_spec_z
    ax1.set_title(fr"{anom_ztfid} ({anom_spec_cls}, $z$={anom_spec_z})" , pad=25)
    plt.xlabel('MJD')
    ax1.set_ylabel('Magnitude')
    ax2.set_ylabel('Probability (%)')

    if anom_idx_is == True: ax1.legend(loc='upper right', ncol=5, bbox_to_anchor=(1.1,1.12), columnspacing=0.45, frameon=False, fontsize=14)
    else: ax1.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.03,1.12), frameon=False, fontsize=14)
    ax2.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.87,1.12), frameon=False, fontsize=14)

    ax1.grid(True)
    ax2.grid(True)

    if savefig:
        plt.savefig(f"{figure_path}/{anom_ztfid}_AD_run_timeseries.pdf", dpi=300, bbox_inches='tight')

    plt.show()


def extract_lc_and_host_features(ztf_id_ref, use_lc_for_ann_only_bool, show_lc=False, show_host=True, host_features=[]):
    start_time = time.time()
    ztf_id_ref = ztf_id_ref #'ZTF20aalxlis' #'ZTF21abmspzt'
    df_path = "../timeseries"

    #try:
    ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztf_id_ref)
    df_ref = ref_info.timeseries.to_pandas()
    #except:
    #    print("antares_client can't find this object. Skip! Continue...")
    #    return

    df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

    try:
        mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
        mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag
    except:
        print(f"No obs for {ztf_id_ref}. pass!\n")
        return

    if show_lc:
        fig, ax = plt.subplots(figsize=(7,7))
        plt.gca().invert_yaxis()

        ax.errorbar(x=df_ref_r.ant_mjd, y=df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r',
                    label=f'REF: {ztf_id_ref}')
        ax.errorbar(x=df_ref_g.ant_mjd, y=df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g')
        plt.show()

    min_obs_count=4

    lightcurve = ref_info.lightcurve
    #print("lightcurve", lightcurve)
    feature_names, property_names, features_count = create_base_features_class(MAGN_EXTRACTOR, FLUX_EXTRACTOR)


    g_obs = list(get_detections(lightcurve, 'g').ant_mjd.values)
    r_obs = list(get_detections(lightcurve, 'R').ant_mjd.values)
    mjd_l = sorted(g_obs+r_obs)

    lc_properties_d_l = []
    len_det_counter_r,len_det_counter_g = 0,0

    band_lc = lightcurve[(~lightcurve['ant_mag'].isna())]
    idx = ~MaskedColumn(band_lc['ant_mag']).mask
    all_detections = remove_simultaneous_alerts(band_lc[idx])
    for ob, mjd in enumerate(mjd_l): # requires 4 obs
        # do time evolution of detections - in chunks

        detections_pb = all_detections[all_detections['ant_mjd'].values <= mjd]
        #print(detections)
        lc_properties_d={}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb['ant_passband'] == band]

            # Ensure locus has >3 obs for calculation
            if (len(detections) < min_obs_count):
                continue
            #print(detections)

            t = detections['ant_mjd'].values
            m = detections['ant_mag'].values
            merr = detections['ant_magerr'].values
            flux = np.power(10.0, -0.4 * m)
            fluxerr = 0.5 * flux * (np.power(10.0, 0.4 * merr) - np.power(10.0, -0.4 * merr))

            magn_features = MAGN_EXTRACTOR(
                t,
                m,
                merr,
                fill_value=None,
            )
            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            #print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
                #if name == "feature_amplitude_magn_g": print(m, value, band)
            #print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {ztf_id_ref}. pass!\n")
        return
    print(f"Extracted LC features for {ztf_id_ref}!")

    end_time = time.time()
    print(f"Extracted LC features in {(end_time - start_time):.2f}s!")

    if not use_lc_for_ann_only_bool:

        # Get GHOST features
        ra,dec=np.mean(df_ref.ant_ra),np.mean(df_ref.ant_dec)
        snName=[ztf_id_ref, ztf_id_ref]
        snCoord = [SkyCoord(ra*u.deg, dec*u.deg, frame='icrs'), SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                hosts = getTransientHosts_with_timeout(transientName=snName, transientCoord=snCoord, GLADE=True, verbose=0,
                                      starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)
            except:
                print(f"GHOST error for {ztf_id_ref}. Retry without GLADE. \n")
                hosts = getTransientHosts_with_timeout(transientName=snName, transientCoord=snCoord, GLADE=False, verbose=0,
                                      starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)

        if len(hosts) > 1:
            hosts = pd.DataFrame(hosts.loc[0]).T

        hosts_df = hosts[host_features]
        hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

        if len(hosts_df) < 1:
            # if any features are nan, we can't use as input
            print(f"Some features are NaN for {ztf_id_ref}. Skip!\n")
            return

        if show_host:
            print(f'http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color')

        hosts_df = hosts[host_features]
        hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

        lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
        lc_and_hosts_df = lc_and_hosts_df.set_index('ztf_object_id')
        lc_and_hosts_df['raMean'] = hosts.raMean.values[0]
        lc_and_hosts_df['decMean'] = hosts.decMean.values[0]
        if not os.path.exists(df_path):
            print(f"Creating path {df_path}.")
            os.makedirs(df_path)
        lc_and_hosts_df.to_csv(f'{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv')

    else:
        print("Saving for lc timeseries only")
        lc_properties_df = lc_properties_df.set_index('ztf_object_id')
        lc_properties_df.to_csv(f'{df_path}/{lc_properties_df.index[0]}_timeseries.csv')

    print(f"Saved results for {ztf_id_ref}!\n")

def extract_lc_and_host_features_YSE_snana_format(IAU_name, ztf_id_ref, yse_lightcurve, ra, dec, show_lc=False, show_host=False):
    IAU_name = IAU_name
    df_path = "../timeseries"

    min_obs_count=4

    lightcurve = yse_lightcurve
    feature_names, property_names, features_count = create_base_features_class(MAGN_EXTRACTOR, FLUX_EXTRACTOR)

    g_obs = list(yse_lightcurve[yse_lightcurve.FLT == "g"].MJD)
    r_obs = list(yse_lightcurve[yse_lightcurve.FLT == "R"].MJD)
    mjd_l = sorted(g_obs+r_obs)

    lc_properties_d_l = []
    len_det_counter_r,len_det_counter_g = 0,0

    all_detections = yse_lightcurve
    for ob, mjd in enumerate(mjd_l): # requires 4 obs
        # do time evolution of detections - in chunks
        detections_pb = all_detections[all_detections["MJD"].values <= mjd]
        #print(detections)
        lc_properties_d={}
        for band, names in property_names.items():
            detections = detections_pb[detections_pb["FLT"] == band]

            # Ensure locus has >3 obs for calculation
            if (len(detections) < min_obs_count):
                continue
            #print(detections)

            t = detections['MJD'].values
            m = detections['MAG'].values
            merr = detections['MAGERR'].values
            flux = detections['FLUXCAL'].values
            fluxerr = detections['FLUXCALERR'].values

            try:
                magn_features = MAGN_EXTRACTOR(
                    t,
                    m,
                    merr,
                    fill_value=None,
                )
            except:
                print(f"{IAU_name} is maybe not sorted?")
                return

            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            # After successfully calculating features, set locus properties and tag
            lc_properties_d["obs_num"] = int(ob)
            lc_properties_d["mjd_cutoff"] = mjd
            lc_properties_d["ztf_object_id"] = ztf_id_ref
            #print(band, m)
            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
                #if name == "feature_amplitude_magn_g": print(m, value, band)
            #print("%%%%%%%%")
        lc_properties_d_l.append(lc_properties_d)

    lc_properties_d_l = [d for d in lc_properties_d_l if d]
    lc_properties_df = pd.DataFrame(lc_properties_d_l)
    if len(lc_properties_df) == 0:
        print(f"Not enough obs for {IAU_name}. pass!\n")
        return
    print(f"Extracted LC features for {IAU_name}/{ztf_id_ref}!")

    # Get GHOST features
    ra,dec=float(ra),float(dec)
    snName=[IAU_name, IAU_name]
    snCoord = [SkyCoord(ra*u.deg, dec*u.deg, frame='icrs'), SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            hosts = getTransientHosts(transientName=snName, snCoord=snCoord, GLADE=True, verbose=0,
                                  starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)
        except:
            print(f"GHOST error for {IAU_name}. Retry without GLADE. \n")
            hosts = getTransientHosts(transientName=snName, snCoord=snCoord, GLADE=False, verbose=0,
                                  starcut='gentle', ascentMatch=False, savepath=tmp, redo_search=False)

    if len(hosts) > 1:
        hosts = pd.DataFrame(hosts.loc[0]).T

    hosts_df = hosts[host_features]
    hosts_df = hosts_df[~hosts_df.isnull().any(axis=1)]

    if len(hosts_df) < 1:
        # if any features are nan, we can't use as input
        print(f"Some features are NaN for {IAU_name}. Skip!\n")
        return

    if show_host:
        print(f'http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={hosts.raMean.values[0]}+{hosts.decMean.values[0]}&filter=color')

    hosts_df = hosts[host_features]
    hosts_df = pd.concat([hosts_df] * len(lc_properties_df), ignore_index=True)

    lc_and_hosts_df = pd.concat([lc_properties_df, hosts_df], axis=1)
    lc_and_hosts_df = lc_and_hosts_df.set_index('ztf_object_id')
    lc_and_hosts_df['raMean'] = hosts.raMean.values[0]
    lc_and_hosts_df['decMean'] = hosts.decMean.values[0]
    lc_and_hosts_df.to_csv(f'{df_path}/{lc_and_hosts_df.index[0]}_timeseries.csv')

    print(f"Saved results for {IAU_name}/{ztf_id_ref}!\n")

def panstarrs_image_filename(position,image_size=None, filter=None):
    """Query panstarrs service to get a list of image names
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :size : int: cutout image size in pixels.
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :filename: str: file name of the cutout
    """

    service = 'https://ps1images.stsci.edu/cgi-bin/ps1filenames.py'
    url = (f'{service}?ra={position.ra.degree}&dec={position.dec.degree}'
           f'&size={image_size}&format=fits&filters={filter}')

    filename_table = pd.read_csv(url, delim_whitespace=True)['filename']
    return filename_table[0] if len(filename_table) > 0 else None


def panstarrs_cutout(position, filename, image_size=None, filter=None):
    """
    Download Panstarrs cutout from their own service
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    if filename:
        service = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?'
        fits_url = (f'{service}ra={position.ra.degree}&dec={position.dec.degree}'
                    f'&size={image_size}&format=fits&red={filename}')
        fits_image = fits.open(fits_url)
    else:
        fits_image = None

    return fits_image


def host_pdfs(ztfid_ref, df, figure_path, ann_num, save_pdf=True, imsizepix=100, change_contrast=False):
    ref_name = df.ZTFID[0]
    data = df

    if save_pdf:
        pdf_path = f'{figure_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.pdf'
        pdf_pages = PdfPages(pdf_path)

    total_plots = len(df)
    rows = 3  # Number of rows in the subplot grid
    cols = 3  # Number of columns in the subplot grid
    num_subplots = rows * cols  # Total number of subplots in each figure
    num_pages = math.ceil(total_plots / num_subplots)

    for page in range(num_pages):
        fig, axs = plt.subplots(rows, cols, figsize=(6, 6))

        for i in range(num_subplots):
            index = page * num_subplots + i

            if index >= total_plots:
                break

            d = df.iloc[index]
            ax = axs[i // cols, i % cols]
            ax.set_xticks([])
            ax.set_yticks([])

            try:  # Has host assoc
                sc = SkyCoord(d['HOST_RA'], d['HOST_DEC'], unit=u.deg)

                outfilename = f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits"

                if os.path.isfile(outfilename):
                    print(f"Remove previously saved cutout {d['ZTFID']}_pscutout.fits to download a new one")
                    os.remove(outfilename)

                if not os.path.exists(outfilename):
                    filename = panstarrs_image_filename(sc, image_size=imsizepix, filter='r')
                    fits_image = panstarrs_cutout(sc, filename, image_size=imsizepix, filter='r')
                    fits_image.writeto(outfilename)


                wcs = WCS(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                imdata = fits.getdata(f"../ps1_cutouts/{d['ZTFID']}_pscutout.fits")

                if change_contrast:
                    transform = AsinhStretch() + PercentileInterval(93)
                else:
                    transform = AsinhStretch() + PercentileInterval(99.5)

                bfim = transform(imdata)
                ax.imshow(bfim, cmap="gray", origin="lower")
                ax.set_title(f"{d['ZTFID']}", pad=0.1, fontsize=18)

            except:
                # Use a red square image when there is no data
                imdata = Image.new('RGB', (100, 100), color = (255, 0, 0)) # red
                ax.imshow(imdata, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f" ", pad=0.1, fontsize=18)

        # Remove axes labels
        for ax in axs.flat:
            ax.label_outer()
            ax.set_xticks([])
            ax.set_yticks([])

        # Reduce padding between subplots for a tighter layout
        plt.tight_layout(pad=0.1)

        plt.ion()
        plt.show()

        if save_pdf:
            pdf_pages.savefig(fig, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.show()

        plt.close(fig)

    if save_pdf:
        pdf_pages.close()
        print(f"PDF saved at: {pdf_path}")

import os
#a silly hack to switch the working directory to the one where this file is located
working_dir = '/root/LAISS/AD_slackbot/'
scripts_dir = os.path.join(working_dir, 'scripts')

import time
import datetime
import math
import numpy as np
import pandas as pd
import sys 
import astro_ghost
import os 
import subprocess

os.chdir(working_dir)
sys.path.append(scripts_dir)

from helper_functions import *
from laiss_functions import *
import requests
from requests.auth import HTTPBasicAuth
import warnings
warnings.filterwarnings("ignore")
import shap
from auth import toku
import astro_ghost

host_path = os.path.join(working_dir, "data/host_info")
if not os.path.exists(host_path):
    os.makedirs(host_path)

os.environ['GHOST_PATH'] = host_path

# Use dummy host-galaxy association database
getGHOST(real=False,verbose=False)

def dict_merge(dict_list):
    dict1 = dict_list[0]
    for dict in dict_list[1:]:
        dict1.update(dict)
    return(dict1)

def plot_shap(ANTARESID, shap_features):
    #lc features
    lc_descripts =  {'feature_amplitude_magn':'Half amplitude of magnitude',
              'feature_anderson_darling_normal_magn': 'Unbiased Anderson–Darling normality test statistic for magnitude',
              'feature_beyond_1_std_magn' : 'Fraction of observations n = 1-sigma beyond the mean magnitude ⟨m⟩',
              'feature_beyond_2_std_magn' : 'Fraction of observations n = 2-sigma beyond the mean magnitude ⟨m⟩.',
              'feature_cusum_magn' : 'A range of cumulative sums dependent on the number of observations, mean magnitude, and magnitud standard deviation',
              'feature_inter_percentile_range_2_magn' : 'Inter-percentile range for p = 0.02, where p is the pth quantile of the magnitude distribution',
              'feature_inter_percentile_range_10_magn' : 'Inter-percentile range for p = 0.10, where p is the pth quantile of the magnitude distribution',
              'feature_inter_percentile_range_25_magn' : 'Inter-percentile range for p = 0.25, where p is the pth quantile of the magnitude distribution',
              'feature_kurtosis_magn' : 'Excess kurtosis of magnitude',
              'feature_linear_fit_slope_magn' : ' The slope of the light curve in the least squares fit of the linear stochastic model with Gaussian noise described by observation errors',
              'feature_linear_fit_slope_sigma_magn' : ' The err on the slope of the light curve in the least squares fit of the linear stochastic model with Gaussian noise described by observation errors',
              'feature_magnitude_percentage_ratio_40_5_magn' : 'The magnitude 40 to 5 ratio, written in terms of the magnitude distribution quantile function Q',
              'feature_magnitude_percentage_ratio_20_5_magn' : 'The magnitude 20 to 5 ratio, written in terms of the magnitude distribution quantile function Q',
              'feature_mean_magn' : 'The non-weighted mean magnitude',
              'feature_median_absolute_deviation_magn' : 'The median of the absolute value of the difference between magnitude and its median',
              'feature_percent_amplitude_magn' : 'The maximum deviation of magnitude from its median',
              'feature_median_buffer_range_percentage_10_magn' : 'Fraction of observations inside Median(m)±10×(max(m)-min(m))/2 interval',
              'feature_median_buffer_range_percentage_20_magn' : 'Fraction of observations inside Median(m)±20×(max(m)-min(m))/2 interval',
              'feature_percent_difference_magnitude_percentile_5_magn' : 'Ratio of p=5th inter-percentile range to the median',
              'feature_percent_difference_magnitude_percentile_10_magn' : 'Ratio of p=10th inter-percentile range to the median',
              'feature_skew_magn' : 'Skewness of magnitude',
              'feature_standard_deviation_magn' : r"Standard deviation of magnitude, $\sigma_m$",
              'feature_stetson_k_magn' : 'Stetson K coefficient described light curve shape of magnitude',
              'feature_weighted_mean_magn' : 'Weighted mean magnitude',
              'feature_anderson_darling_normal_flux' : ' Unbiased Anderson–Darling normality test statistic for flux',
              'feature_cusum_flux' : 'A range of cumulative sums dependent on the number of observations, mean flux, and flux standard deviation',
              'feature_excess_variance_flux' : 'Measure of the flux variability amplitude',
              'feature_kurtosis_flux' : ' Excess kurtosis of flux',
              'feature_mean_variance_flux' : 'Standard deviation of flux to mean flux ratio',
              'feature_skew_flux' : 'Skewness of flux',
              'feature_stetson_k_flux' : 'Stetson K coefficient described light curve shape of flux'}

    #add all g and r-band features here
    lc_descripts_bands = {}

    for key, val in lc_descripts.items():
        for band in 'gr':
            lc_descripts_bands[key + f'_{band}'] = val + f", in {band}"

    #host features
    host_descripts =  {'momentXX':'Host second-order moment MXX',
                   'momentXY':'Host second-order moment MXY',
                   'momentYY':'Host second-order moment MYY',
                   'momentR1':'Host first radial moment',
                   'momentRH':'Host half radial moment',
                   'PSFFlux':'Host PSF flux',
                   'ApFlux':'Host aperture flux',
                   'KronFlux':'Host kron flux',
                   'KronRad':'Host kron radius',
                   'ExtNSigma':'Host extendedness'}

    #add all g and r-band features here
    host_descripts_bands = {}
    for key, val in host_descripts.items():
        for band in 'grizy':
            host_descripts_bands[band + key] = val + f", in {band}"

    host_descripts_nonGeneric = {'i-z':'Host i-z',
                             'gApMag_gKronMag':'Host aperture mag minus kron mag (shape param) in g',
                             'rApMag_rKronMag':'Host aperture mag minus kron mag (shape param) in r',
                             'iApMag_iKronMag':'Host aperture mag minus kron mag (shape param) in i',
                             'zApMag_zKronMag':'Host aperture mag minus kron mag (shape param) in z',
                             'yApMag_yKronMag':'Host aperture mag minus kron mag (shape param) in y',
                             '4DCD':'Host 4D distance from Tonry stellar locus',
                             '7DCD':'Host 4D distance from Tonry stellar locus',
                             'dist/DLR':'DLR-normalized angular offset'}

    full_descript_dict = dict_merge([lc_descripts_bands, host_descripts_bands, host_descripts_nonGeneric])


    # Hyperparameters for best AD model
    n_estimators = 100
    max_depth = 35
    random_state = 11
    max_features = 35

    with open(os.path.join(working_dir, "data", "host_features.txt")) as host_f:
        host_features = [line.strip() for line in host_f.readlines()]

    with open(os.path.join(working_dir, "data", "lc_features.txt")) as lc_f:
        lc_features = [line.strip() for line in lc_f.readlines()]

    lc_and_host_features = lc_features + host_features

    model_path = os.path.join(working_dir, f"models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model")

    with open(f'{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl', 'rb') as f:
        clf = pickle.load(f)

    RFdata = pd.read_csv(os.path.join(working_dir, "data", "dataset_bank_tns_df_resampled_train_SMOTE_train_test_70-30_min14_kneighbors8.csv.gz"))
    explainer = shap.TreeExplainer(clf, data=RFdata[lc_and_host_features])

    lc_and_hosts_df = pd.DataFrame(shap_features, index=[0])
    chosen_instance = lc_and_hosts_df[lc_and_host_features].iloc[[-1]]
    shap_values = explainer.shap_values(chosen_instance)
    fig = shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], chosen_instance, matplotlib=True, show=False, text_rotation=15, label_dict=full_descript_dict);
    plt.title(f"Force Plot for {ANTARESID}\n\n\n\n", fontsize=16, fontweight='bold')
    fig.patch.set_edgecolor('k')
    filepath = os.path.join(working_dir, "plots", f"forcePlots/{ANTARESID}_ForcePlot.png")
    plt.savefig(filepath,dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3, transparent=False, edgecolor=fig.get_edgecolor());
    print(f"File successfully saved at {filepath}.")
    return filepath

def upload_file(token, file_path, channel):
    url = "https://slack.com/api/files.upload"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    files = {
        'file': open(file_path, 'rb')
    }
    data = {
        "channels": channel,
        "initial_comment": "Here is the file you requested.",
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

def make_file_public(token, file_id):
    url = "https://slack.com/api/files.sharedPublicURL"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "file": file_id
    }
    response = requests.post(url, headers=headers, data=data)
    print(response)
    return response.json()

def share_public_link(token, channel_id, file_id):
    # Get the public URL from the file response
    url = "https://slack.com/api/files.info"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "file": file_id
    }
    response = requests.get(url, headers=headers, params=params)
    public_url = response.json()['file']['permalink_public']
    return public_url

def format_url(url, filename):
    team_id = url.split("/")[-1].split("-")[0]
    file_id = url.split("/")[-1].split("-")[1]
    pub_secret = url.split("/")[-1].split("-")[2]

    formatted_url = f"https://files.slack.com/files-pri/{team_id}-{file_id}/{filename.lower()}?pub_secret={pub_secret}"
    return formatted_url
  

def shap_values(ANTARESID, shap_values, channelID):
    plotpath = plot_shap(ANTARESID, shap_values)
    if plotpath:
        filename = plotpath.split("/")[-1]
        response = upload_file(toku, plotpath, channelID)
        print(response)
        file_id = response['file']['id']
        make_file_public(toku, file_id)
        initial_url = share_public_link(toku, channelID, file_id)
        final_url = format_url(initial_url, filename)
        print("Final URL is...")
        print(final_url)
        return final_url
    else:
        return ''

import requests as req
import os
import pandas as pd
from astro_ghost.PS1QueryFunctions import geturl
from auth import toku
import json
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

working_dir = '/root/LAISS/AD_slackbot/'

data_dir = os.path.join(working_dir, 'data')
log_dir = os.path.join(working_dir, 'logs')

class build_rec:
    def __init__(self, antaresID, tns_name, tns_cls, anom_score, brightest_mag, brightest_phase, first_mag, first_phase, host_name, host_sep, host_photoz, host_z_flag, ra, dec, ztfname, shap_url, posted_before=False):
        '''
        Builds a string and dataframe row for easy posting. Also can post to slack.

        Parameters
        ----------
        TODO: update!

        Parameters
        ----------
        TODO: update!
        '''
        if tns_name == "No TNS":
            self.ziggy_url = 'None'
            self.antares_url = f'https://antares.noirlab.edu/loci/{antaresID}'
        else:
            self.ziggy_url = f'https://ziggy.ucolick.org/yse/transient_detail/{tns_name}'
            self.antares_url = f'https://antares.noirlab.edu/loci/{antaresID}'

        self.name = antaresID
        self.tns_name = tns_name
        self.tns_cls = tns_cls
        self.anom_score = anom_score
        self.brightest_mag = brightest_mag
        self.brightest_phase = brightest_phase
        self.first_phase = first_phase
        self.first_mag = first_mag
        self.posted_before = posted_before
        self.host_name = host_name
        self.host_sep = host_sep
        self.host_photoz = host_photoz
        self.ztfname = ztfname
        self.shap_url = shap_url
        self.host_z_flag = host_z_flag[0] + host_z_flag[1:].lower()
        if (self.host_photoz > 0):
            self.brightest_abs_mag = brightest_mag - cosmo.distmod(self.host_photoz).value
            self.first_abs_mag = first_mag - cosmo.distmod(self.host_photoz).value

            d_A = cosmo.angular_diameter_distance(z=self.host_photoz)
            theta = self.host_sep*u.arcsec
            self.host_sep_kpc = (theta * d_A).to(u.kpc, u.dimensionless_angles())
        else:
            self.brightest_abs_mag = np.nan
            self.first_abs_mag = np.nan
            self.host_sep_kpc = np.nan
        self.ra = ra
        self.dec = dec

        #pic of the field
        if self.dec > -30:
            self.ps1_pic = geturl(self.ra, self.dec, color=True)
        else:
            self.ps1_pic = ""

def post(bs_set=None, string=None, lookback_t=1, today_mjd=None, channel=None):
        '''
    Posts to a slack channel. If no string is provided, will use the string attribute of the object. This is a standalone function for automation purposes.

    Parameters
    ----------
    string : str, optional
        String to post to slack. If None, will use the string attribute of the object.
    channel : str, optional
        Channel to post to. Specific to workspace the bot token has been installed in.
        '''
        if string is None:
            raise ValueError('No string provided')

        if len(string) < 1:
            ps = f"No candidates with an alert in the last {lookback_t} day (MJD {today_mjd - lookback_t} - {today_mjd})."
            p1=req.post('https://slack.com/api/chat.postMessage',
                 json={'channel':channel,
                         'text':ps,
                         'mrkdwn':'true',
                         'parse':'none'},
                         headers={'Authorization': f'Bearer {toku}'})
            p1.raise_for_status()
        else:
            initialMessage = string[0] #f"Long-duration (>100d) anomalous candidates for today:\n"
            #the header
            p1=req.post('https://slack.com/api/chat.postMessage',
                     json={'channel':channel,
                         'text':initialMessage,
                         'mrkdwn':'true',
                         'parse':'none'},
                         headers={'Authorization': f'Bearer {toku}'})

            string = string[1:]
            attachments = []
 
            reactions = pd.read_csv(os.path.join(data_dir, "slack_reactions.csv"))
            reactions.reset_index(drop=True, inplace=True)

            for i in np.arange(len(bs_set)):
                transient = bs_set[i]

                title = f':collision: {transient.name} :collision:'
                title_link = transient.antares_url
               
                #shap_url = shap_values(transient.ztfname, '')

                if not transient.posted_before: 
                    color = "#36a64f"
                else: 
                    color = "#6D2E9E"

                attachment = {
                    "color": color,
                    "title": title,
                    "title_link": title_link,
                    "fields": [
                    {"title":"TNS Name", "value":f"<{transient.ziggy_url}|{transient.tns_name}>", "short":True},
                    {"title":"TNS Spec. Class", "value":transient.tns_cls, "short":True},
                    {"title":f"Anomaly Score", "value":f"{int(round(transient.anom_score, 1))}%", "short":True},
                    ],
                "callback_id": f"transient_{transient.name}",
                "actions": [
                    {
                    "name": f"badan_{transient.name}",
                    "text": " :yesagn: ",
                    "type": "button",
                    "value": "AGN"
                    },
                    #{
                    #"name": f"goodan_{transient.name}",
                    #"text": " :noagn: ",
                    #"type": "button",
                    #"value": "Not an AGN"
                    #}
                    {
                    "name": f"goodan_{transient.name}",
                    "text": " :thumbsup: ",
                    "type": "button",
                    "value": "Anomalous"
                    },
                    {
                    "name": f"greatan_{transient.name}",
                    "text": " :thumbsup::thumbsup: ",
                    "type": "button",
                    "value": "Very Anomalous"
                    },
                    {
                    "name":f"badan_{transient.name}",
                    "text": " :thumbsdown: ",
                    "type": "button",
                    "value": "Not Anomalous"
                    }
                ]
                }

                if (transient.brightest_abs_mag == transient.brightest_abs_mag):
                    attachment['fields'].append({"title":f"First Detection (From {transient.host_z_flag}-z)", "value":f"{transient.first_mag:.2f}/{transient.first_abs_mag:.2f}, {transient.first_phase:.1f}d ago.", "short": True})
                    attachment['fields'].append({"title":f"Peak Magnitude (From {transient.host_z_flag}-z)", "value":f"{transient.brightest_mag:.2f}/{transient.brightest_abs_mag:.2f}, {transient.brightest_phase:.1f}d ago.", "short": True})
                else:
                    attachment['fields'].append({"title":"First Detection", "value":f"{transient.first_mag:.2f}, {transient.first_phase:.1f}d ago.", "short": True})
                    attachment['fields'].append({"title":f"Peak App. Magnitude", "value":f"{transient.brightest_mag:.2f}, {transient.brightest_phase:.1f}d ago.", "short": True})
                if transient.tns_name == 'No TNS':
                    attachment['fields'][0] = {"title":"TNS Name", "value":f"---", "short":True}
                if transient.host_sep >= 0:
                    attachment['fields'].append({"title":"GHOST Host", "value":transient.host_name, "short": True})
                    attachment['fields'].append({"title":f"Host Separation (From {transient.host_z_flag}-z)", "value":f"{transient.host_sep:.0f}\'\'/{transient.host_sep_kpc:.0f}", "short":True})
                    attachment['fields'].append({"title":f"{transient.host_z_flag}-z", "value":f"{transient.host_photoz:.4f}", "short":True})
                else:
                    attachment['fields'].append({"title":"GHOST Host", "value":"No Host Found", "short":True})

                url = transient.ps1_pic
                shap_url = transient.shap_url
                if len(url) > 1:
                    attachment['image_url'] = url
                #if len(url) > 1:
                #    attachment['fields'].append({'value': f'<{url}|PS1 Field Stamp>', 'short': False})
                if len(shap_url) > 1:
                    attachment['fields'].append({'value': f'<{shap_url}|Shap Force Plot>', 'short': False})

                NameStr = "";

                #get all people who have marked an event as anomalous
                #some slightly more sophisticated handling to consider only the latest tag someone has listed 
                idx = reactions.groupby(['User','Transient'])['TimeStamp'].idxmax()

                # Subset the DataFrame using these indices
                latest_reactions = reactions.loc[idx]
                
                df_flagged = latest_reactions.loc[(latest_reactions['Transient'] == transient.name) & (latest_reactions['Response'].isin(['Anomalous', 'Very Anomalous']))]
                flaggedusers = np.unique(df_flagged['UserID'].values)
                if len(flaggedusers) > 0:
                    NameStr += ", ".join([f"<@{x}>" for x in flaggedusers]) + " marked this event as anomalous.\n"
 
                df_notflagged = latest_reactions.loc[(latest_reactions['Transient'] == transient.name) & (latest_reactions['Response'].isin(['Not Anomalous']))]
                notflaggedusers = np.unique(df_notflagged['UserID'].values)
                
                if len(notflaggedusers) > 0: 
                    NameStr += ", ".join([f"<@{x}>" for x in notflaggedusers]) + " marked this event as not anomalous."

                df_flagged = latest_reactions.loc[(latest_reactions['Transient'] == transient.name) & (latest_reactions['Response'].isin(['AGN']))]
                flaggedusers = np.unique(df_flagged['UserID'].values)
                if len(flaggedusers) > 0:
                    NameStr += ", ".join([f"<@{x}>" for x in flaggedusers]) + " marked this event as an AGN.\n"

                df_notflagged = latest_reactions.loc[(latest_reactions['Transient'] == transient.name) & (latest_reactions['Response'].isin(['Not an AGN']))]
                notflaggedusers = np.unique(df_notflagged['UserID'].values)

                if len(notflaggedusers) > 0:
                    NameStr += ", ".join([f"<@{x}>" for x in notflaggedusers]) + " marked this event as not an AGN."

                print(NameStr)

                if len(NameStr) > 0:
                    attachment['fields'].append({
                            "title": "",
                            "value": NameStr,
                            "short": 'false'
                        })

                attachments.append(attachment)

            p1=req.post('https://slack.com/api/chat.postMessage',
                json={'channel':channel,
                 'mrkdwn':'true',
                 'parse':'none',
                 'attachments': json.dumps(attachments)},
                 headers={'Authorization': f'Bearer {toku}'})

            p1.raise_for_status()

            # Additional check for Slack-specific errors in the response body
            response_data = p1.json()
            if not response_data.get('ok'):
                raise ValueError(f"Failed to post message: {response_data.get('error')}")

        if 200 <= p1.status_code < 300:
            print('Posted to Slack successfully.')

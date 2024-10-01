import requests as req
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

channel = 'C03STCB0ACA'
slackbot_pwd = os.environ['SLACKBOT_PWD']
p1=req.get('https://slack.com/api/conversations.history',
params={'channel':channel, 'parse':'none', 'limit':999},
 headers={'Authorization': f'Bearer {slackbot_pwd}'})

p1.raise_for_status()

# Additional check for Slack-specific errors in the response body
response_data = p1.json()
if not response_data.get('ok'):
    raise ValueError(f"Failed to check posting history: {response_data.get('error')}")

posted_names = []
posted_ts = []
posted_AGNvotes = []
posted_notAGNvotes = []
posted_anomvotes = []
posted_notanomvotes = []
posted_ts = []

for message in response_data['messages']:
    try:
        names = [x['title'].split(" ")[1] for x in message['attachments']]
        reactions = ([x['fields'][-1] for x in message['attachments']])
        reactions = [x['value'] for x in reactions]
        for i in np.arange(len(names)):
            oneSetReactions = reactions[i].split(".")
            AGNvote = np.nansum(["as an AGN" in x for x in oneSetReactions] )
            notAGNvote = np.nansum(["not an AGN" in x for x in oneSetReactions])
            anomalyvote = np.nansum(["as anomalous" in x for x in oneSetReactions])
            notanomalyvote = np.nansum(["not anomalous" in x for x in oneSetReactions])
            
            posted_names.append(names[i])
            posted_AGNvotes.append(AGNvote)
            posted_notAGNvotes.append(notAGNvote)
            posted_anomvotes.append(anomalyvote)
            posted_notanomvotes.append(notanomalyvote)
            
        #set up matrix
        ts = [message['ts']]*len(names)
        posted_ts.append(ts)
 
    except:
        pass

df = pd.DataFrame({'Transient':np.array(posted_names), 'TimeStamp':np.concatenate(posted_ts), 'AGN Votes':np.array(posted_AGNvotes), 'Not AGN Votes':np.array(posted_notAGNvotes), 
'Anomaly Votes':np.array(posted_anomvotes), 'Not Anomaly Votes':np.array(posted_notanomvotes)})

print("Number of newly labeled AGN:")
print(len(df[df['AGN Votes'] >= 1]))

print("Number of AGN-like interesting transients:")
print(len(df[df['Not AGN Votes'] >= 1]))

df.to_csv("ActiveLabelingResults_AGN2024.csv",index=False)

# Get the current datetime
#current_datetime = datetime.now()

# Subtract 5 days
#five_days_ago = current_datetime - timedelta(days=20)
# If you need it as a timestamp (e.g., for use in APIs)
#timestamp_five_days_ago = five_days_ago.timestamp()
#df = df[df['TimeStamp'].astype('float') > timestamp_five_days_ago]

#the names of the transients posted <numDays ago
#print(df['Transient'].values)

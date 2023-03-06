import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import scienceplots

plt.style.use("science")


all_data = pd.read_csv("data/processed/1997-01-01-2022-12-31.csv")
data = pd.read_csv("data/processed/long.csv")

######
# DATA FROM PISKORSKI ET AL. (2020)
######
# dataset can be found here: piskorski.waw.pl/resources/acled/ACLED-DATASETS.zip
piskorski_data = pd.read_csv(
    "data/raw/ACLED-I.txt",
    delimiter="\t",
    header=None,
    names=["text", "country", "class"],
)

sub_event_types = {
    "sub-event type": piskorski_data["class"].value_counts(),
    "percentage": piskorski_data["class"].value_counts() / len(piskorski_data) * 100,
}
piskorski_classes = pd.DataFrame(sub_event_types)

study_sub_events = {
    "sub-event type": data["sub_event_type"].value_counts(),
    "percentage": data["sub_event_type"].value_counts() / len(data) * 100,
}
study_classes = pd.DataFrame(study_sub_events)


# #####
# # CLASS DISTRIBUTION
# #####

# Sunburst of the class distribution
fig = px.sunburst(data, path=["event_type", "sub_event_type"])
fig.show()

# event type
event_types = {
    "event type": data["event_type"].value_counts(),
    "percentage": data["event_type"].value_counts() / len(data) * 100,
}
event_type_distribution = pd.DataFrame(event_types)

# sub-event type
sub_event_types = {
    "sub-event type": data["sub_event_type"].value_counts(),
    "percentage": data["sub_event_type"].value_counts() / len(data) * 100,
}
sub_event_type_distribution = pd.DataFrame(sub_event_types)


# Visualization of event type counts
f, ax = plt.subplots(figsize=(5, 4))
sns.countplot(
    x=data["event_type"],
    order=[
        "Battles",
        "Explosions/Remote violence",
        "Violence against civilians",
        "Protests",
        "Riots",
        "Strategic developments",
    ],
    palette=[
        (27 / 255, 210 / 255, 243 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        (255 / 255, 161 / 255, 90 / 255, 1),
        (2 / 255, 204 / 255, 150 / 255, 1),
        (239 / 255, 85 / 255, 59 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
    ],
)
ax.set(xlabel="Event types", ylabel="count")
ax.tick_params(axis="x", rotation=90)
plt.setp(
    ax,
    xticklabels=[
        "Battles",
        "Explosions/ \n Remote violence",
        "Violence against \n civilians",
        "Protests",
        "Riots",
        "Strategic \n developments",
    ],
)
plt.legend([], [], frameon=False)
plt.show()

# Visualization - count plot for sub-event Types
f, ax = plt.subplots(figsize=(5, 4))
sns.countplot(
    x=data["sub_event_type"],
    order=[
        "Armed clash",
        "Government regains territory",
        "Non-state actor overtakes territory",
        "Chemical weapon",
        "Air/drone strike",
        "Suicide bomb",
        "Shelling/artillery/missile attack",
        "Remote explosive/landmine/IED",
        "Grenade",
        "Sexual violence",
        "Attack",
        "Abduction/forced disappearance",
        "Peaceful protest",
        "Protest with intervention",
        "Excessive force against protesters",
        "Violent demonstration",
        "Mob violence",
        "Agreement",
        "Arrests",
        "Change to group/activity",
        "Disrupted weapons use",
        "Headquarters or base established",
        "Looting/property destruction",
        "Non-violent transfer of territory",
        "Other",
    ],
    palette=[
        (27 / 255, 210 / 255, 243 / 255, 1),
        (27 / 255, 210 / 255, 243 / 255, 1),
        (27 / 255, 210 / 255, 243 / 255, 1),
        #
        (99 / 255, 110 / 255, 251 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        (99 / 255, 110 / 255, 251 / 255, 1),
        #
        (255 / 255, 161 / 255, 90 / 255, 1),
        (255 / 255, 161 / 255, 90 / 255, 1),
        (255 / 255, 161 / 255, 90 / 255, 1),
        #
        (2 / 255, 204 / 255, 150 / 255, 1),
        (2 / 255, 204 / 255, 150 / 255, 1),
        (2 / 255, 204 / 255, 150 / 255, 1),
        #
        (239 / 255, 85 / 255, 59 / 255, 1),
        (239 / 255, 85 / 255, 59 / 255, 1),
        #
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
        (171 / 255, 99 / 255, 250 / 255, 1),
    ],
)
# ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set(xlabel="Subvent types")
ax.tick_params(axis="x", rotation=90)
plt.setp(
    ax,
    xticklabels=[
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
    ],
)
plt.legend([], [], frameon=False)
plt.show()

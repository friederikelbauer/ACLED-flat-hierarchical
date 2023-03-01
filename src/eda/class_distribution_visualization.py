import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import plotly.express as px
import numpy as np
import scienceplots

plt.style.use("science")


all_data = pd.read_csv("data/processed/1997-01-01-2022-12-31.csv")
data = pd.read_csv("data/processed/long.csv")

######
# PISKORSKI ET AL.
######
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


# #####
# # CLASS DISTRIBUTION
# #####

# Sunburst of the class distribution
# fig = px.sunburst(data, path=["event_type", "sub_event_type"])
# fig.show()

# # event type
# event_types = {
#     "event type": data["event_type"].value_counts(),
#     "percentage": data["event_type"].value_counts() / len(data) * 100,
# }
# event_type_distribution = pd.DataFrame(event_types)

# sub-event type
# sub_event_types = {
#     "sub-event type": data["sub_event_type"].value_counts(),
#     "percentage": data["sub_event_type"].value_counts() / len(data) * 100,
# }
# sub_event_type_distribution = pd.DataFrame(sub_event_types)


# Visualization of event type counts
f, ax = plt.subplots(figsize=(2, 4))
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
    palette=["#4A4A4A"],
)
ax.set(xlabel="Event types", ylabel="count")
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
    ],
)
plt.legend([], [], frameon=False)
plt.show()


# Visualization of event types in pie format
# event_counts = Counter(data["event_type"])

# plt.pie(
#     [float(v) for v in event_counts.values()],
#     labels=[k for k in event_counts],
#     autopct=None,
# )
# plt.show()

# # Visualization of sub-event types as pie chart
"""sources: 
1) https://stackoverflow.com/questions/49199164/increasing-pie-chart-size-with-matplotlib-radius-parameter-appears-to-do-nothin
2) https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html#sphx-glr-gallery-color-colormap-reference-py
"""
# counts = Counter(data["sub_event_type"])
# sizes = [float(v) for v in counts.values()]
# labels = [k for k in counts]
# fig1, ax1 = plt.subplots(figsize=(8, 7))
# fig1.subplots_adjust(0.3, 0, 1, 1)
# theme = plt.get_cmap("hsv")
# ax1.set_prop_cycle("color", [theme(1.0 * i / len(sizes)) for i in range(len(sizes))])
# _, _ = ax1.pie(sizes, startangle=90)
# ax1.axis("equal")
# total = sum(sizes)
# plt.legend(
#     loc="upper left",
#     labels=[
#         "%s, %1.1f%%" % (l, (float(s) / total) * 100) for l, s in zip(labels, sizes)
#     ],
#     prop={"size": 11},
#     bbox_to_anchor=(0.0, 1),
#     bbox_transform=fig1.transFigure,
# )
# plt.show()

# # Visualization - count plot for sub-event Types
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

# # Visualization 5
# f, ax = plt.subplots(figsize=(7, 5))
# sns.histplot(
#     data,
#     x="sub_event_type",
#     hue="event_type",
#     multiple="stack",
#     edgecolor=".3",
#     linewidth=0.5,
#     palette="rocket",
# )
# # ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
# ax.set(
#     xlabel="Sub-event Types",
#     # yscale="log"
# )
# ax.tick_params(axis="x", rotation=90)
# plt.legend([], [], frameon=False)

# plt.show()

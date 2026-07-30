#importing packages
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import t
import os
from scipy import stats

#import data
households_2022 = pd.read_csv("../NHTS_2022/Data/hhv2pub.csv")
trips_2022      = pd.read_csv("../NHTS_2022/Data/tripv2pub.csv")
households_2017 = pd.read_csv("../NHTS_2017/Data/hhpub.csv")
trips_2017      = pd.read_csv("../NHTS_2017/Data/trippub.csv")
households_2009 = pd.read_csv("../NHTS_2009/Data/HHV2PUB.csv")
trips_2009      = pd.read_csv("../NHTS_2009/Data/DAYV2PUB.csv")
households_2001 = pd.read_csv("../NHTS_2001/Data/HHPUB.csv")
trips_2001      = pd.read_csv("../NHTS_2001/Data/DAYPUB.csv")

households = [households_2022, households_2017, households_2009, households_2001]
trips = [trips_2022, trips_2017, trips_2009, trips_2001]
survey_years = [2022, 2017, 2009, 2001]

#relabeling data
life_cycle_order = [
    '2+ adults, with children',
    'Adults, no children',
    'Retired adults, no children',
    "Single father, with children",
    "Single mother, with children"
]
lif_cyc_labels_abbr = {
    "1": "Adults, no children", "2": "Adults, no children", "3": "Single adult, with children",
    "4": "2+ adults, with children",  "5": "Single adult, with children",   "6": "2+ adults, with children",
    "7": "Single adult, with children","8": "2+ adults, with children", "9": "Retired adults, no children", "10": "Retired adults, no children"
}

WHYTRP1S_labels = {
    "1": "Home",
    "10": "Work",
    "20": "School/Daycare/Religious",
    "30": "Medical/Dental services",
    "40": "Shopping/Errands",
    "50": "Social/Recreational",
    "60": "Other",
    "70": "Transport someone",
    "80": "Meals",
    "97": "Other"
}

WHYTRP1S_labels_2001 = {
    "1": "Work",
    "2": "Work",
    "3": "School/Daycare/Religious",
    "4": "School/Daycare/Religious",
    "5": "Medical/Dental services",
    "6": "Shopping/Errands",
    "7": "Other",
    "8": "Social/Recreational",
    "9": "Meals",
    "10": "Transport someone",
    "11": "Work",
    "12": "Home",
    "13": "Other"
}
HHFAMINC_2017_label = {
    '1': 'Less than $25,000',  
    '2': 'Less than $25,000', 
    '3': 'Less than $25,000',
    '4': '$25,000 to $49,999',
    '5': '$25,000 to $49,999',
    '6': '$50,000 to $74,999',
    '7': '$75,000 to $99,999',
    '8': '$100,000 or more',
    '9': '$100,000 or more',
    '10': '$100,000 or more',
    '11': '$100,000 or more',
    '-7': None,
    '-8': None,
    '-9': None
}
HHFAMINC_2022_label = {
    '1': 'Less than $25,000',  
    '2': 'Less than $25,000', 
    '3': 'Less than $25,000',
    '4': '$25,000 to $49,999',
    '5': '$25,000 to $49,999',
    '6': '$50,000 to $74,999',
    '7': '$75,000 to $99,999',
    '8': '$100,000 or more',
    '9': '$100,000 or more',
    '10': '$100,000 or more',
    '11': '$100,000 or more',
    '-7': None,
    '-8': None,
    '-9': None
}
HHFAMINC_2009_label = {
    '1': 'Less than $25,000',  
    '2': 'Less than $25,000', 
    '3': 'Less than $25,000',
    '4': 'Less than $25,000',
    '5': 'Less than $25,000',
    '6': '$25,000 to $49,999',
    '7': '$25,000 to $49,999',
    '8': '$25,000 to $49,999',
    '9': '$25,000 to $49,999',
    '10': '$25,000 to $49,999',
    '11': '$50,000 to $74,999',
    '12': '$50,000 to $74,999',
    '13': '$50,000 to $74,999',
    '14': '$50,000 to $74,999',
    '15': '$50,000 to $74,999',
    '16': '$75,000 to $99,999',
    '17': '$75,000 to $99,999',
    '18': '$100,000 or more',
    '-7': None,
    '-8': None,
    '-9': None
}
HHFAMINC_2001_label = {
    '1': 'Less than $25,000',  
    '2': 'Less than $25,000', 
    '3': 'Less than $25,000',
    '4': 'Less than $25,000',
    '5': 'Less than $25,000',
    '6': '$25,000 to $49,999',
    '7': '$25,000 to $49,999',
    '8': '$25,000 to $49,999',
    '9': '$25,000 to $49,999',
    '10': '$25,000 to $49,999',
    '11': '$50,000 to $74,999',
    '12': '$50,000 to $74,999',
    '13': '$50,000 to $74,999',
    '14': '$50,000 to $74,999',
    '15': '$50,000 to $74,999',
    '16': '$75,000 to $99,999',
    '17': '$75,000 to $99,999',
    '18': '$100,000 or more',
    '-7': None,
    '-8': None,
    '-9': None
}

TRPTRANS_2017_label = {
    # Walk / Bike
    1: 4,  # Walk
    2: 5,  # Bike
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    10: 3,
    7: 6,
    8: 6,
    9: 6,
    17: 6,
    18: 6,
    19: 6,
    20: 6,
    -7: None,
    -8: None,
    -9: None
}
TRIPMODE_labels = {
    1: "Privately Owned Vehicle",
    2: "Public Transit",
    3: "School Bus",
    4: "Walk",
    5: "Bike",
    6: "Other"
}

TRPTRANS_2009_to_TRIPMODE_map = {
    1: 1,   # Car
    2: 1,   # Van
    3: 1,   # SUV
    4: 1,   # Pickup truck
    5: 1,   # Other truck
    22: 5,  # Bicycle
    23: 4,  # Walk
    11: 3,  # School bus
    9: 2,   # Local public bus
    10: 2,  # Commuter bus
    12: 2,  # Charter/tour bus
    13: 2,  # City-to-city bus
    14: 2,  # Shuttle bus
    15: 2,  # Amtrak/inter city train
    16: 2,  # Commuter train
    17: 2,  # Subway/elevated train
    18: 2,  # Street car/trolley
    6: 6,   # RV
    7: 6,   # Motorcycle
    8: 6,   # Light electric vehicle (golf cart)
    19: 6,  # Taxicab
    20: 6,  # Ferry
    21: 6,  # Airplane
    24: 6,  # Special transit - people with disabilities
    97: 6,  # Other
    -1: None,
    -7: None,
    -8: None,
    -9: None
}

TRPTRANS_2001_to_TRIPMODE_map = {
    1: 1,   # Car
    2: 1,   # Van
    3: 1,   # SUV
    4: 1,   # Pickup truck
    5: 1,   # Other truck
    25: 5,  # Bicycle
    26: 4,  # Walk
    12: 3,  # School bus
    10: 2,   # Local public bus
    11: 2,  # Commuter bus
    13: 2,  # Charter/tour bus
    14: 2,  # City-to-city bus
    15: 2,  # Amtrak/inter city train
    16: 2,  # Commuter train
    17: 2,  # Subway/elevated train
    18: 2,  # Street car/trolley
    6: 6,   # RV
    7: 6,   # Motorcycle
    8: 6,   # Commercial/charter airplane
    9: 6, #Private/corporate airplane
    19: 6,  # Ship/Cruise
    20: 6,  # Passenger Line/Ferry
    21: 6,  # Sailboat/Yacht
    22: 6,  #taxicab
    23: 6, #limo
    24: 6,  # Hotel/airport shuttle
    91: 6,  # Other
    -1: None,
    -7: None,
    -8: None,
    -9: None
}

TRIPMODE_2022_labels = {
    1: "Privately Owned Vehicle",          # Previously: Privately Owned Vehicle - Driver
    2: "Privately Owned Vehicle",          # Previously: Privately Owned Vehicle - Passenger
    3: "Public Transit",
    4: "School Bus",
    5: "Walk",
    6: "Bike",
    7: "Other",
    -9: None,
    -8: None,
    -7: None
}

sex_labels = {
    "1": "Male",
    "2": "Female",
}

HHFAMINC_LABEL_MAPS = {
    2022: HHFAMINC_2022_label,
    2017: HHFAMINC_2017_label,
    2009: HHFAMINC_2009_label,
    2001: HHFAMINC_2001_label,
}

TRIPMODE_LABEL_MAPS = {
    2022: TRIPMODE_2022_labels,
    2017: TRPTRANS_2017_label,
    2009: TRPTRANS_2009_to_TRIPMODE_map,
    2001: TRPTRANS_2001_to_TRIPMODE_map,
}


#relabels and cleans the trips data
for i, df in enumerate(trips):
    #set surveyyear
    df["surveyyear"] = survey_years[i]

    #set combine R_SEX_IMP and R_SEX into R_SEX_std
    if "R_SEX_IMP" in df.columns:
        df["R_SEX"] = df["R_SEX"].replace([-7, -8], pd.NA)
        df["R_SEX_std"] = df["R_SEX_IMP"].fillna(df["R_SEX"])
    else:
        df["R_SEX_std"] = df["R_SEX"].replace([-7, -8], pd.NA)

    df["R_SEX_std"] = df["R_SEX_std"].astype(float)
    
    
    df["HOUSEID"] = df["HOUSEID"].astype(str)

    #set ref_sex so we can distinguish single parent household gender
    ref_sex = (
        df.loc[
            df["PERSONID"] == 1,
            ["HOUSEID", "R_SEX_std"]
        ]
        .drop_duplicates(subset="HOUSEID", keep="first")
        .rename(columns={"R_SEX_std": "REF_SEX_std"})
        .copy()
    )

    ref_sex["surveyyear"] = survey_years[i]
    
    #merge in the reference sex
    df["HOUSEID"] = df["HOUSEID"].astype(str)

    df = df.merge(
        ref_sex,
        on=["HOUSEID", 'surveyyear'],
        how="left",
        validate="m:1"
    )

    #label life_cycle groups
    df["LIF_CYC_label_abbr"] = (
        df["LIF_CYC"].astype(str).map(lif_cyc_labels_abbr)
    )

    single_mask = df["LIF_CYC_label_abbr"] == "Single adult, with children"

    df.loc[
        single_mask & (df["REF_SEX_std"] == 1),
        "LIF_CYC_label_abbr"
    ] = "Single father, with children"

    df.loc[
        single_mask & (df["REF_SEX_std"] == 2),
        "LIF_CYC_label_abbr"
    ] = "Single mother, with children"

    df["LIF_CYC_label_abbr"] = df["LIF_CYC_label_abbr"].replace(
        "Single adult, with children", np.nan
    )

    df = df.dropna(subset=['LIF_CYC_label_abbr'])

    #label WHYTRP1S
    year= survey_years[i]
    
    if year == 2001:
        df["WHYTRP1S_labeled"] = (
            df["WHYTRP1S"]
            .astype(str)
            .map(WHYTRP1S_labels_2001)
        )
    else:
        df["WHYTRP1S_labeled"] = (
            df["WHYTRP1S"]
            .astype(str)
            .map(WHYTRP1S_labels)
        )
    
    
    #label HHFAMINC
    df["HHFAMINC_label"] = (
        df["HHFAMINC"]
        .astype(str)
        .map(HHFAMINC_LABEL_MAPS[year])
    )

    #fix for matplotlib
    df["HHFAMINC_label"] = (
        df["HHFAMINC_label"]
        .astype("string")
        .str.replace("$", r"\$", regex=False)
    )
    
    #label tripmode
    if year == 2022:
        df["TRIPMODE_label"] = df["TRIPMODE"].map(TRIPMODE_LABEL_MAPS[year])
    else:
        df["TRIPMODE_label"] = (
            df["TRPTRANS"].map(TRIPMODE_LABEL_MAPS[year])
        ).map(TRIPMODE_labels)
    
    #drop trips heading home
    df=df[df['WHYTRP1S_labeled']!='Home']
    df = df.dropna(subset=['LIF_CYC_label_abbr']) 
    trips[i] = df
    
#create metrics
# Combine all years into one dataframe
trips_all = pd.concat(trips, ignore_index=True)

#make sure all needed columns are present

tier_cols = [
    'LIF_CYC_label_abbr',
    'TRIPMODE_label',
    'WHYTRP1S_labeled'
]

for col in tier_cols:
    if col not in trips_all.columns:
        trips_all[col] = np.nan

# create person_miles
#basically multiplying the number of trip miles by the number of people on the trip


trips_all['person_miles'] = trips_all['TRPMILES'] * trips_all['NUMONTRP']

#identify and remove outliers

def weighted_quantile(values, weights, q):
    values = np.asarray(values)
    weights = np.asarray(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    return values[np.cumsum(weights) >= q * weights.sum()][0]

p99_miles = weighted_quantile(
    trips_all['TRPMILES'].dropna(),
    trips_all.loc[trips_all['TRPMILES'].notna(), 'WTTRDFIN'],
    0.99
)

p99_time = weighted_quantile(
    trips_all['TRVLCMIN'].dropna(),
    trips_all.loc[trips_all['TRVLCMIN'].notna(), 'WTTRDFIN'],
    0.99
)

trips_trimmed = trips_all[
    (trips_all['TRPMILES'] <= p99_miles) &
    (trips_all['TRVLCMIN'] <= p99_time)
].copy()

# weighting functions

def weighted_median(values, weights):
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    return values[np.cumsum(weights) >= weights.sum() / 2][0]


def weighted_mode(values, weights):
    return (
        pd.DataFrame({'v': values, 'w': weights})
        .groupby('v')['w']
        .sum()
        .idxmax()
    )


def weighted_stats(group, value_col):
    g = group[[value_col, 'WTTRDFIN']].dropna()
    if g.empty:
        return pd.Series({'mean': np.nan, 'median': np.nan, 'mode': np.nan})

    v = g[value_col].to_numpy()
    w = g['WTTRDFIN'].to_numpy()

    return pd.Series({
        'mean': np.average(v, weights=w),
        'median': weighted_median(v, w),
        'mode': weighted_mode(v, w)
    })


def trip_and_occupancy_stats(group):
    """Weighted trip count + mean occupants"""
    g = group[['WTTRDFIN', 'NUMONTRP']].dropna()
    if g.empty:
        return pd.Series({
            'TRIPS': np.nan,
            'NUMONTRP_mean': np.nan
        })

    return pd.Series({
        'TRIPS': g['WTTRDFIN'].sum(),  # weighted number of trips
        'NUMONTRP_mean': np.average(g['NUMONTRP'], weights=g['WTTRDFIN'])
    })

# define tiers

tiers = {
    'tier_1': ['surveyyear', 'LIF_CYC_label_abbr'],
    'tier_2': ['surveyyear', 'LIF_CYC_label_abbr', 'TRIPMODE_label'],
    'tier_3': ['surveyyear', 'LIF_CYC_label_abbr', 'TRIPMODE_label', 'WHYTRP1S_labeled']
}

# generate metrics

frames = []

for tier_name, cols in tiers.items():

    miles = (
        trips_trimmed
        .groupby(cols, dropna=False)
        .apply(weighted_stats, value_col='TRPMILES')
        .reset_index()
        .rename(columns=lambda c: f'TRPMILES_{c}' if c in ['mean','median','mode'] else c)
    )

    time = (
        trips_trimmed
        .groupby(cols, dropna=False)
        .apply(weighted_stats, value_col='TRVLCMIN')
        .reset_index()
        .rename(columns=lambda c: f'TRVLCMIN_{c}' if c in ['mean','median','mode'] else c)
    )

    pm = (
        trips_trimmed
        .groupby(cols, dropna=False)
        .apply(weighted_stats, value_col='person_miles')
        .reset_index()
        .rename(columns=lambda c: f'PERSON_MILES_{c}' if c in ['mean','median','mode'] else c)
    )

    trips_occ = (
        trips_trimmed
        .groupby(cols, dropna=False)
        .apply(trip_and_occupancy_stats)
        .reset_index()
    )

    out = (
        miles
        .merge(time, on=cols)
        .merge(pm, on=cols)
        .merge(trips_occ, on=cols)
    )

    out['tier'] = tier_name
    frames.append(out)


nhts_trip_metrics = pd.concat(frames, ignore_index=True)

#clean up columns
nhts_trip_metrics=nhts_trip_metrics[['Survey Year', 'Life Cycle Group', 'Trip Mode',
                    'Trip Purpose','Mean People on Trip',
                'Mean Trip Minutes', 'Median Trip Minutes', 'Mode Trip Minutes', 'Mean Trip Miles', 'Median Trip Miles',
       'Mode Trip Miles','Mean Person Miles', 'Median Person Miles', 'Mode Person Miles',
       'Tier']]


nhts_trip_metrics.to_csv('../Outputs/Tables/Core_Data_Analysis_Minutes_Miles.csv', index=False)

#isolate tier 1 for plotting
nhts_trip_tier1 = nhts_trip_metrics[nhts_trip_metrics['tier']=='tier_1']

#Chart trip length over time
plt.figure(figsize=(10, 6))

# Plot one line for each life-cycle group
for group, subset in nhts_trip_tier1.groupby("LIF_CYC_label_abbr"):

    # Sort by year to ensure lines connect correctly
    subset = subset.sort_values("surveyyear")

    plt.plot(
        subset["surveyyear"],
        subset["TRPMILES_mean"],
        marker="o",
        label=group
    )

plt.xticks(
    [2001, 2009, 2017, 2022],
    [2001, 2009, 2017, 2022]
)
    
# Labels and title
plt.xlabel("Survey Year")
plt.ylabel("Mean Trip Miles")
plt.title("Mean Trip Distance by Life Cycle Group Over Time")

# Show legend
plt.legend(
    title="Life Cycle Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# Improve layout
plt.tight_layout()
plt.savefig(
    "../Outputs/Charts/avg_trips_distance_by_life_cycle_year.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

#chart trip duration over time
plt.figure(figsize=(10, 6))

# Plot one line for each life-cycle group
for group, subset in nhts_trip_tier1.groupby("LIF_CYC_label_abbr"):

    # Sort by year to ensure lines connect correctly
    subset = subset.sort_values("surveyyear")

    plt.plot(
        subset["surveyyear"],
        subset["TRVLCMIN_mean"],
        marker="o",
        label=group
    )

plt.xticks(
    [2001, 2009, 2017, 2022],
    [2001, 2009, 2017, 2022]
)
    
# Labels and title
plt.xlabel("Survey Year")
plt.ylabel("Mean Trip Duration (Minutes)")
plt.title("Mean Trip Duration by Life Cycle Group Over Time")

# Show legend
plt.legend(
    title="Life Cycle Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# Improve layout
plt.tight_layout()
plt.savefig(
    "../Outputs/Charts/avg_trips_duration_by_life_cycle_year.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(10, 6))

# Plot one line for each life-cycle group
for group, subset in nhts_trip_tier1.groupby("LIF_CYC_label_abbr"):

    # Sort by year to ensure lines connect correctly
    subset = subset.sort_values("surveyyear")

    plt.plot(
        subset["surveyyear"],
        subset["NUMONTRP_mean"],
        marker="o",
        label=group
    )

plt.xticks(
    [2001, 2009, 2017, 2022],
    [2001, 2009, 2017, 2022]
)
    
# Labels and title
plt.xlabel("Survey Year")
plt.ylabel("Mean number of people on trips")
plt.title("Mean Trip Passengers by Life Cycle Group Over Time")

# Show legend
plt.legend(
    title="Life Cycle Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# Improve layout
plt.tight_layout()
plt.savefig(
    "../Outputs/Charts/avg_trips_passengers_by_life_cycle_year.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


#getting household sizes
#reading in persons data to get head of household sex
#sex is contained in trip and persons, not household
persons_2022    = pd.read_csv("../NHTS_2022/Data/perv2pub.csv")
persons_2017    = pd.read_csv("../NHTS_2017/Data/perpub.csv")
persons_2009    = pd.read_csv("../NHTS_2009/Data/PERV2PUB.csv")
persons_2001    = pd.read_csv("../NHTS_2001/Data/PERPUB.csv")

persons = [persons_2022, persons_2017, persons_2009, persons_2001]

#standardizing sex
for i, df in enumerate(persons):
    if "R_SEX_IMP" in df.columns:
        df["R_SEX"] = df["R_SEX"].replace([-7, -8], pd.NA)
        df["R_SEX_std"] = df["R_SEX_IMP"].fillna(df["R_SEX"])
    else:
        df["R_SEX_std"] = df["R_SEX"].replace([-7, -8], pd.NA)
    persons[i] = df

# isolating head of household
refpersons_list = []
for i, persons_df in enumerate(persons):
    persons_df["HOUSEID"] = persons_df["HOUSEID"].astype(str)
    ref_sex = (
        persons_df.loc[persons_df["PERSONID"] == 1, ["HOUSEID", "R_SEX_std"]]
        .drop_duplicates(subset="HOUSEID", keep="first")
        .copy()
    )
    ref_sex["surveyyear"] = survey_years[i]
    refpersons_list.append(ref_sex)

# merging gender of head of household into household data
for i, hh_df in enumerate(households):
    hh_df["surveyyear"] = survey_years[i]
    hh_df["HOUSEID"] = hh_df["HOUSEID"].astype(str)
    hh_df = hh_df.merge(refpersons_list[i], on=["HOUSEID", 'surveyyear'], how="left", validate="m:1")
    households[i] = hh_df

#relabeling life cycle groups in households
for i, df in enumerate(households):

    #label life_cycle groups
    df["LIF_CYC_label_abbr"] = (
        df["LIF_CYC"].astype(str).map(lif_cyc_labels_abbr)
    )

    single_mask = df["LIF_CYC_label_abbr"] == "Single adult, with children"

    df.loc[
        single_mask & (df["R_SEX_std"] == 1),
        "LIF_CYC_label_abbr"
    ] = "Single father, with children"

    df.loc[
        single_mask & (df["R_SEX_std"] == 2),
        "LIF_CYC_label_abbr"
    ] = "Single mother, with children"

    df["LIF_CYC_label_abbr"] = df["LIF_CYC_label_abbr"].replace(
        "Single adult, with children", np.nan
    )
    households[i] = df
    
hh_all = pd.concat(households, ignore_index=True)

#generating household counts
hh_counts = (
    hh_all
    .groupby(['surveyyear', 'LIF_CYC_label_abbr'], dropna=False)
    ['HOUSEID']
    .nunique()
    .reset_index(name='n_households')
)

print(hh_counts)

#generating average household size
avg_hh_size = (
    hh_all.dropna(
        subset=[
            "surveyyear",
            "LIF_CYC_label_abbr",
            "HHSIZE",
            "WTHHFIN"
        ]
    )
    .groupby(
        ["surveyyear", "LIF_CYC_label_abbr"]
    )
    .apply(
        lambda x: np.average(
            x["HHSIZE"],
            weights=x["WTHHFIN"]
        )
    )
    .reset_index(name="HHSIZE_mean")
)

#pivoting avg_hh_size to a more readable format
avg_hh_size_wide = (
    avg_hh_size
    .pivot(
        index="LIF_CYC_label_abbr",
        columns="surveyyear",
        values="HHSIZE_mean"
    )
    .sort_index(axis=1)
)
print(avg_hh_size_wide)

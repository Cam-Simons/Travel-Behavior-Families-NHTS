#importing packages
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import t
import os

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

#applying relabeling and some data cleaning

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
    df=df[df['WHYTRP1S_labeled']!='Home']
    df = df.dropna(subset=['LIF_CYC_label_abbr']) 
    
    trips[i] = df
    

##Functions for plotting and statistical testing
def weighted_ttests_full(
    df,
    value_cols=("trips_per_capita", "pct_share"),
    weight_col="WTHHFIN",
    group_var="LIF_CYC_label_abbr",
    category_col="TRIPMODE_label", #defaults to tripmode, can be changed to WHYTRP1S_labeled by setting the category col in the function command
    year_col="surveyyear",
    household_col="HOUSEID",
    min_eff_n=5,
    var_epsilon=1e-12,
    se_epsilon=1e-12,
    verbose=False
):

    # Convert tuple to list
    value_cols = list(value_cols)
    
    #helper functions

    def clean_subset(x, w):

        mask = (
            x.notna()
            & w.notna()
            & (w > 0)
            & np.isfinite(x)
            & np.isfinite(w)
        )

        return x[mask], w[mask]


    def weighted_stats(x, w):

        x, w = clean_subset(x, w)

        if len(x) < 2:
            return np.nan, np.nan, np.nan

        w_sum = w.sum()

        if w_sum <= 0:
            return np.nan, np.nan, np.nan

        # Weighted mean
        mean = np.average(
            x,
            weights=w
        )

        # Weighted population variance
        var = np.average(
            (x - mean) ** 2,
            weights=w
        )

        # Kish effective sample size
        n_eff = (
            w_sum ** 2 /
            (w ** 2).sum()
        )

        return mean, var, n_eff


    # clean df

    df = df.copy()

    # Remove invalid life-cycle groups
    df = df[
        df[group_var].notna()
    ]

    df = df[
        df[group_var].astype(str).str.strip() != ""
    ]

    df = df[
        df[group_var] != "None"
    ]


    # Define expected categories
    #this will help account for 0s iin a category in a year later

    expected_categories = df[category_col].dropna().unique().tolist()



    #  Household × category × year grid
    #this is to flesh out any possible missing categories

    # Unique household-level information
    households = (
        df[
            [
                household_col,
                group_var,
                year_col,
                weight_col
            ]
        ]
        .drop_duplicates()
    )


    # Check for duplicate household/group/year combinations
    duplicate_check = (
        households
        .duplicated(
            subset=[
                household_col,
                group_var,
                year_col
            ],
            keep=False
        )
    )

    if duplicate_check.any():

        if verbose:
            print(
                "Warning: duplicate household/group/year "
                "combinations detected."
            )


    # Create all mode combinations for every household
    modes = pd.DataFrame({
        category_col: expected_categories
    })


    households["_merge_key"] = 1
    modes["_merge_key"] = 1

    complete_grid = (
        households
        .merge(
            modes,
            on="_merge_key",
            how="left"
        )
        .drop(
            columns="_merge_key"
        )
    )


    # Merge Observed values into grid

    observed_cols = [
        household_col,
        group_var,
        year_col,
        category_col
    ]

    observed_values = df[
        observed_cols +
        value_cols
    ].copy()


    # Ensure there is only one observation per household,
    # life-cycle group, year, and mode.
    duplicate_values = (
        observed_values
        .duplicated(
            subset=observed_cols,
            keep=False
        )
    )

    if duplicate_values.any():

        raise ValueError(
            "The dataframe contains multiple rows for the same "
            "HOUSEID × LIF_CYC_label_abbr × surveyyear × "
            "TRIPMODE_label combination. "
            "Aggregate these rows before running this function."
        )


    df_complete = (
        complete_grid
        .merge(
            observed_values,
            on=observed_cols,
            how="left"
        )
    )


    # Fill missing values with 0

    # These are combinations where there was no observed row.
    # They represent zero trips for this household/mode/year.
    # IMPORTANT:
    # We only fill values with zero when the entire row is missing.
    # Existing NaNs in an observed row remain NaN.

    row_was_missing = (
        df_complete[value_cols]
        .isna()
        .all(axis=1)
    )

    for metric in value_cols:

        df_complete.loc[
            row_was_missing,
            metric
        ] = 0


    # clean weights

    df_complete = df_complete[
        df_complete[weight_col].notna()
    ]

    df_complete = df_complete[
        df_complete[weight_col] > 0
    ]


    # Testing

    results = []


    for metric in value_cols:


        # between_year 
        # Same life-cycle group
        # Same trip mode
        # Different survey years

        for (
            group,
            cat
        ), subset in df_complete.groupby(
            [
                group_var,
                category_col
            ]
        ):

            years = sorted(
                subset[
                    year_col
                ]
                .dropna()
                .unique()
            )

            if len(years) < 2:
                continue


            for y1, y2 in combinations(
                years,
                2
            ):

                s1 = subset[
                    subset[year_col] == y1
                ]

                s2 = subset[
                    subset[year_col] == y2
                ]


                mean1, var1, n1_eff = weighted_stats(
                    s1[metric],
                    s1[weight_col]
                )

                mean2, var2, n2_eff = weighted_stats(
                    s2[metric],
                    s2[weight_col]
                )


                # Guards
                if (
                    np.isnan(mean1)
                    or np.isnan(mean2)
                    or np.isnan(var1)
                    or np.isnan(var2)
                    or n1_eff < min_eff_n
                    or n2_eff < min_eff_n
                ):
                    if verbose:
                        print(
                            "Skipped sparse cell:",
                            group,
                            cat,
                            y1,
                            y2
                        )
                    continue

                # Standard error

                se1 = np.sqrt(
                    var1 /
                    n1_eff
                )

                se2 = np.sqrt(
                    var2 /
                    n2_eff
                )

                se_diff = np.sqrt(
                    se1 ** 2 +
                    se2 ** 2
                )


                if se_diff < se_epsilon:
                    continue


                # Welch t-statistic

                t_stat = (
                    mean1 -
                    mean2
                ) / se_diff


                # Welch-Satterthwaite degrees of freedom

                df_num = (
                    se1 ** 2 +
                    se2 ** 2
                ) ** 2

                df_den = (
                    (se1 ** 4) /
                    (n1_eff - 1)
                    +
                    (se2 ** 4) /
                    (n2_eff - 1)
                )

                df_w = (
                    df_num /
                    df_den
                    if df_den > 0
                    else np.nan
                )


                # Two-sided p-value

                p_val = (
                    2 *
                    (
                        1 -
                        t.cdf(
                            abs(t_stat),
                            df_w
                        )
                    )
                    if np.isfinite(df_w)
                    else np.nan
                )


                results.append({

                    "test_type":
                        "between_years",

                    "metric":
                        metric,

                    group_var:
                        group,

                    category_col:
                        cat,

                    "year_1":
                        y1,

                    "year_2":
                        y2,

                    "mean_1":
                        mean1,

                    "mean_2":
                        mean2,

                    "mean_diff":
                        mean1 - mean2,

                    "t_stat":
                        t_stat,

                    "degrees_of_freedom":
                        df_w,

                    "p_value":
                        p_val,

                    "significant":
                        (
                            p_val < 0.05
                            if pd.notna(p_val)
                            else False
                        ),

                    "effective_sample_size_1":
                        n1_eff,

                    "effective_sample_size_2":
                        n2_eff
                })


        # within_year 
        #
        # Different life-cycle groups
        # Same survey year
        # Same trip mode

        for (
            year,
            cat
        ), subset in df_complete.groupby(
            [
                year_col,
                category_col
            ]
        ):

            groups = sorted(
                subset[
                    group_var
                ]
                .dropna()
                .unique()
            )

            if len(groups) < 2:
                continue


            for g1, g2 in combinations(
                groups,
                2
            ):

                s1 = subset[
                    subset[group_var] == g1
                ]

                s2 = subset[
                    subset[group_var] == g2
                ]


                mean1, var1, n1_eff = weighted_stats(
                    s1[metric],
                    s1[weight_col]
                )

                mean2, var2, n2_eff = weighted_stats(
                    s2[metric],
                    s2[weight_col]
                )


                # Guards

                if (
                    np.isnan(mean1)
                    or np.isnan(mean2)
                    or np.isnan(var1)
                    or np.isnan(var2)
                    or n1_eff < min_eff_n
                    or n2_eff < min_eff_n
                ):
                    if verbose:
                        print(
                            "Skipped sparse cell:",
                            group,
                            cat,
                            y1,
                            y2
                        )
                    continue


                # Standard errors

                se1 = np.sqrt(
                    var1 /
                    n1_eff
                )

                se2 = np.sqrt(
                    var2 /
                    n2_eff
                )

                se_diff = np.sqrt(
                    se1 ** 2 +
                    se2 ** 2
                )


                if se_diff < se_epsilon:
                    continue


                # Welch t-statistic

                t_stat = (
                    mean1 -
                    mean2
                ) / se_diff


                # Welch-Satterthwaite degrees of freedom

                df_num = (
                    se1 ** 2 +
                    se2 ** 2
                ) ** 2

                df_den = (
                    (se1 ** 4) /
                    (n1_eff - 1)
                    +
                    (se2 ** 4) /
                    (n2_eff - 1)
                )

                df_w = (
                    df_num /
                    df_den
                    if df_den > 0
                    else np.nan
                )


                # Two-sided p-value

                p_val = (
                    2 *
                    (
                        1 -
                        t.cdf(
                            abs(t_stat),
                            df_w
                        )
                    )
                    if np.isfinite(df_w)
                    else np.nan
                )


                results.append({

                    "test_type":
                        "within_year_groups",

                    "metric":
                        metric,

                    category_col:
                        cat,

                    "year":
                        year,

                    "group_1":
                        g1,

                    "group_2":
                        g2,

                    "mean_1":
                        mean1,

                    "mean_2":
                        mean2,

                    "mean_diff":
                        mean1 - mean2,

                    "t_stat":
                        t_stat,

                    "degrees_of_freedom":
                        df_w,

                    "p_value":
                        p_val,

                    "significant":
                        (
                            p_val < 0.05
                            if pd.notna(p_val)
                            else False
                        ),

                    "effective_sample_size_1":
                        n1_eff,

                    "effective_sample_size_2":
                        n2_eff
                })




    return pd.DataFrame(results)

def plot_multiwave_trips_stacked_household(
    trips_list,
    households_list,
    survey_years,
    group_var,
    stack_var,                     
    stack_label,
    group_order=None,
    var_name=None,
    exclude_home=False,             # redundant now that Home is excluded in the cleaning, shouldn't break anything
    exclude_groups=("I don't know", "I prefer not to answer"),
    outlier_method="percentile",
    outlier_threshold=0.99,
    fixed_max=30,
    zscore_max=3,
    figsize=(18, 7),
    save_dir=None,
    save_prefix=""
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    combined_avgs = []
    combined_long = []

    # loop on surveyyear
    for trips_df, hh_df, year in zip(trips_list, households_list, survey_years):
        trips_df = trips_df.copy()
        hh_df = hh_df.copy()

        # redundant now that home is excluded in cleaning but shouldn't break anything
        if exclude_home and "WHYTRP1S_labeled" in trips_df.columns:
            trips_df = trips_df[trips_df["WHYTRP1S_labeled"] != "Home"]

        # Trip counts by household by group_var by stack_var
        hh_trip = (
            trips_df
            .groupby(["HOUSEID", group_var, stack_var])
            .size()
            .reset_index(name="n_trips")
        )

        # Pivot
        hh_trip = (
            hh_trip
            .pivot_table(
                index=["HOUSEID", group_var],
                columns=stack_var,
                values="n_trips",
                fill_value=0
            )
            .reset_index()
        )

        # Merge WTHHFIN and HHSIZE
        hh_trip["HOUSEID"] = hh_trip["HOUSEID"].astype(str)
        hh_df["HOUSEID"] = hh_df["HOUSEID"].astype(str)

        hh_trip = hh_trip.merge(
            hh_df[["HOUSEID", "WTHHFIN", "HHSIZE"]],
            on="HOUSEID",
            how="left"
        )

        
        # Drop exclude_groups
        if exclude_groups:
            hh_trip = hh_trip[~hh_trip[group_var].isin(exclude_groups)]

            
        # Per-capita 
        stack_cols = [
            c for c in hh_trip.columns
            if c not in ("HOUSEID", group_var, "WTHHFIN", "HHSIZE")
        ]

        for c in stack_cols:
            hh_trip[c] = hh_trip[c] / hh_trip["HHSIZE"]

            
        # exclude outliers on 99th percentiles

        hh_total = hh_trip[stack_cols].sum(axis=1)

        if outlier_method == "percentile":
            hh_trip = hh_trip[hh_total <= hh_total.quantile(outlier_threshold)]
        elif outlier_method == "fixed":
            hh_trip = hh_trip[hh_total <= fixed_max]
        elif outlier_method == "zscore":
            mu, sd = hh_total.mean(), hh_total.std()
            hh_trip = hh_trip[hh_total <= mu + zscore_max * sd]
      
    
        # making long to calculate % share
        hh_long = hh_trip.melt(
            id_vars=["HOUSEID", group_var, "WTHHFIN"],
            value_vars=stack_cols,
            var_name=stack_var,
            value_name="trips_per_capita"
        )

        hh_long["surveyyear"] = year
        hh_long["weighted_trips"] = hh_long["trips_per_capita"] * hh_long["WTHHFIN"]

        total_by_house = (
            hh_long
            .groupby(["surveyyear", "HOUSEID"])["weighted_trips"]
            .transform("sum")
        )

        hh_long["pct_share"] = hh_long["weighted_trips"] / total_by_house
        hh_long.drop(columns="weighted_trips", inplace=True)
        combined_long.append(hh_long)


        # weighted_mean
       
        def weighted_mean(x, w):
            mask = w.notna()
            if mask.sum() == 0:
                return np.nan
            return (x[mask] * w[mask]).sum() / w[mask].sum()

        avg = (
            hh_long
            .groupby([group_var, stack_var])
            .apply(lambda g: weighted_mean(
                g["trips_per_capita"],
                g["WTHHFIN"]
            ))
            .reset_index(name="avg_trips_per_capita")
        )

        avg["surveyyear"] = year
        combined_avgs.append(avg)

    combined_df = pd.concat(combined_avgs, ignore_index=True)
    combined_long_df = pd.concat(combined_long, ignore_index=True)
    
    
    # Plotting section
    display_name = var_name or group_var.replace("_", " ").title()
    year_order = sorted(survey_years)
    groups = group_order or combined_df[group_var].dropna().unique()
    
    # Figure 1: By year, average count
    fig1, axes = plt.subplots(1, len(year_order), figsize=figsize, sharey=True)

    for ax, year in zip(axes, year_order):
        df = combined_df[combined_df["surveyyear"] == year]
        pivot = df.pivot(index=group_var, columns=stack_var,
                         values="avg_trips_per_capita").fillna(0)

        if group_order:
            pivot = pivot.reindex(group_order)

        pivot.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(str(year))
        ax.set_xlabel("")
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig1.legend(handles, labels, title=stack_label,
                loc="lower center", ncol=6)
    fig1.suptitle(f"{display_name}: Trips per Capita", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_dir:
        fig1.savefig(os.path.join(save_dir, f"{save_prefix}ByYear_TripsPerCapita.png"),
                     dpi=300, bbox_inches="tight")
    
    
    plt.show()

    # Figure 2: by year percentage
    fig2, axes = plt.subplots(1, len(year_order), figsize=figsize, sharey=True)

    for ax, year in zip(axes, year_order):
        df = combined_df[combined_df["surveyyear"] == year]
        pivot = df.pivot(index=group_var, columns=stack_var,
                         values="avg_trips_per_capita").fillna(0)
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

        if group_order:
            pivot = pivot.reindex(group_order)

        pivot.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(str(year))
        ax.set_xlabel("")
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")

    fig2.legend(handles, labels, title=stack_label,
                loc="lower center", ncol=6)
    fig2.suptitle(f"{display_name}: Trip Share (%)", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_dir:
        fig2.savefig(os.path.join(save_dir, f"{save_prefix}ByYear_Percent.png"),
                     dpi=300, bbox_inches="tight")
        
        
    plt.show()

    # Figure 3: by group_var count
    fig3, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)

    for ax, grp in zip(axes, groups):
        df = combined_df[combined_df[group_var] == grp]
        pivot = df.pivot(index="surveyyear", columns=stack_var,
                         values="avg_trips_per_capita").loc[year_order]

        pivot.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(grp)
        ax.set_xlabel("")

    fig3.legend(handles, labels, title=stack_label,
                loc="lower center", ncol=6)
    fig3.suptitle(f"{display_name}: Trips per Capita by Group", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_dir:
        fig3.savefig(os.path.join(save_dir, f"{save_prefix}ByGroup_TripsPerCapita.png"),
                     dpi=300, bbox_inches="tight")
        
        
    plt.show()

    # Figure 4: by group_var percent
    fig4, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)

    for ax, grp in zip(axes, groups):
        df = combined_df[combined_df[group_var] == grp]
        pivot = (
            df.pivot(index="surveyyear", columns=stack_var,
                     values="avg_trips_per_capita")
            .loc[year_order]
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

        pivot.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(grp)
        ax.set_xlabel("")

    fig4.legend(handles, labels, title=stack_label,
                loc="lower center", ncol=6)
    fig4.suptitle(f"{display_name}: Trip Share (%) by Group", y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_dir:
        fig4.savefig(os.path.join(save_dir, f"{save_prefix}ByGroup_Percent.png"),
                     dpi=300, bbox_inches="tight")
        
        
    plt.show()

    return combined_df, combined_long_df



#plot Life Cycle Groups by Trip Purpose
combined_df_purpose_no_home, long_df_purpose_no_home = plot_multiwave_trips_stacked_household(
    trips_list=trips,
    households_list=households,
    survey_years=[2022, 2017, 2009, 2001],

    group_var="LIF_CYC_label_abbr",
    stack_var="WHYTRP1S_labeled",
    stack_label="Trip Purpose",

    group_order=[
        "Adults, no children",
        "Single mother, with children",
        "Single father, with children",
        "2+ adults, with children",
        "Retired adults, no children"
    ],

    var_name="Life Cycle",

    exclude_home=True,   # redundant now

    save_dir="../Outputs/Charts/",
    save_prefix="LifeCycle_Purpose_NoHome_"
)

#plotting average daily trips by life cycle group over time
#simplifies the data output above so it's no longer counting trips by trip mode or purpose
simplified_combined_trips=combined_df_purpose_no_home.groupby(['LIF_CYC_label_abbr', 'surveyyear'])['avg_trips_per_capita'].sum().reset_index()
plt.figure(figsize=(10, 6))

# Plot one line for each life-cycle group
for group, subset in simplified_combined_trips.groupby("LIF_CYC_label_abbr"):

    # Sort by year to ensure lines connect correctly
    subset = subset.sort_values("surveyyear")

    plt.plot(
        subset["surveyyear"],
        subset["avg_trips_per_capita"],
        marker="o",
        label=group
    )

plt.xticks(
    [2001, 2009, 2017, 2022],
    [2001, 2009, 2017, 2022]
)
    
# Labels and title
plt.xlabel("Survey Year")
plt.ylabel("Mean number of trips")
plt.title("Mean Number of Trips by Life Cycle Group Over Time")

# Show legend
plt.legend(
    title="Life Cycle Group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# Improve layout
plt.tight_layout()

plt.savefig(
    "../Outputs/Charts/avg_trips_per_capita_by_life_cycle_year.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

results_purpose = weighted_ttests_full(
    long_df_purpose_no_home,
    group_var="LIF_CYC_label_abbr",
    category_col="WHYTRP1S_labeled"
)

#“Effective sample sizes are reported using the Kish approximation to account for unequal household weights.”


results_purpose_between_capita= results_purpose[(results_purpose['test_type']=='between_years') & (results_purpose['metric']=='trips_per_capita')]
results_purpose_within_capita= results_purpose[(results_purpose['test_type']=='within_year_groups') & (results_purpose['metric']=='trips_per_capita')]
results_purpose_between_pct= results_purpose[(results_purpose['test_type']=='between_years') & (results_purpose['metric']=='pct_share')]
results_purpose_within_pct= results_purpose[(results_purpose['test_type']=='within_year_groups') & (results_purpose['metric']=='pct_share')]

results_purpose_between_capita = results_purpose_between_capita.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_purpose_within_capita = results_purpose_within_capita.drop(columns=['test_type', 'metric', 'LIF_CYC_label_abbr', 'year_1', 'year_2'])
results_purpose_between_pct = results_purpose_between_pct.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_purpose_within_pct = results_purpose_within_pct.drop(columns=['test_type', 'metric', 'LIF_CYC_label_abbr', 'year_1', 'year_2'])
results_purpose_between_pct = results_purpose_between_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_purpose_within_pct = results_purpose_within_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_purpose_within_capita = results_purpose_within_capita[['group_1', 'group_2', 'WHYTRP1S_labeled', 'year','mean_1', 'mean_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]
results_purpose_within_pct = results_purpose_within_pct[['group_1', 'group_2', 'WHYTRP1S_labeled', 'year','pct_1', 'pct_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]

results_purpose_between_capita.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripPurpose_between_years_capita.csv', index=False)
results_purpose_within_capita.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripPurpose_within_year_capita.csv', index=False)
results_purpose_between_pct.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripPurpose_between_years_pct.csv', index=False)
results_purpose_within_pct.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripPurpose_within_years_pct.csv', index=False)

combined_df_purpose_no_home.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripPurpose.csv', index=False)


#plotting Household income by Trip purpose
combined_df_income, long_df_income = plot_multiwave_trips_stacked_household(
    trips_list=trips,
    households_list=households,
    survey_years=[2022, 2017, 2009, 2001],

    group_var="HHFAMINC_label",
    stack_var="WHYTRP1S_labeled",
    stack_label="Trip Purpose",

    group_order=[
    "Less than \$25,000",
    "\$25,000 to \$49,999",
    "\$50,000 to \$74,999",
    "\$75,000 to \$99,999",
    "\$100,000 or more"
],

    var_name="Household Income Group",

    exclude_home=True,  

    save_dir="../Outputs/Charts/",
    save_prefix="Household_Income_"
)

results_income = weighted_ttests_full(
    long_df_income,
    group_var="HHFAMINC_label",
    category_col="WHYTRP1S_labeled"
)

results_income_between_capita= results_income[(results_income['test_type']=='between_years') & (results_income['metric']=='trips_per_capita')]
results_income_within_capita= results_income[(results_income['test_type']=='within_year_groups') & (results_income['metric']=='trips_per_capita')]
results_income_between_pct= results_income[(results_income['test_type']=='between_years') & (results_income['metric']=='pct_share')]
results_income_within_pct= results_income[(results_income['test_type']=='within_year_groups') & (results_income['metric']=='pct_share')]

results_income_between_capita = results_income_between_capita.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_income_within_capita = results_income_within_capita.drop(columns=['test_type', 'metric', 'HHFAMINC_label', 'year_1', 'year_2'])
results_income_between_pct = results_income_between_pct.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_income_within_pct = results_income_within_pct.drop(columns=['test_type', 'metric', 'HHFAMINC_label', 'year_1', 'year_2'])
results_income_between_pct = results_income_between_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_income_within_pct = results_income_within_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_income_within_capita = results_income_within_capita[['group_1', 'group_2', 'WHYTRP1S_labeled', 'year','mean_1', 'mean_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]
results_income_within_pct = results_income_within_pct[['group_1', 'group_2', 'WHYTRP1S_labeled', 'year','pct_1', 'pct_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]

results_income_between_capita.to_csv('../Outputs/Tables/IncomeGroup_by_TripPurpose_between_years_capita.csv', index=False)
results_income_within_capita.to_csv('../Outputs/Tables/IncomeGroup_by_TripPurpose_within_year_capita.csv', index=False)
results_income_between_pct.to_csv('../Outputs/Tables/IncomeGroup_by_TripPurpose_between_years_pct.csv', index=False)
results_income_within_pct.to_csv('../Outputs/Tables/IncomeGroup_by_TripPurpose_within_years_pct.csv', index=False)

combined_df_income.to_csv('../Outputs/Tables/IncomeGroup_by_TripPurpose.csv', index=False)

#plotting Life cycle group by trip mode
combined_df_lifecycle_mode, long_df_lifecycle_mode = plot_multiwave_trips_stacked_household(
    trips_list=trips,
    households_list=households,
    survey_years=[2022, 2017, 2009, 2001],

    group_var="LIF_CYC_label_abbr",
    stack_var="TRIPMODE_label",
    stack_label="Trip Mode",

    group_order=[
     "Adults, no children",
        "Single mother, with children",
        "Single father, with children",
        "2+ adults, with children",
        "Retired adults, no children"
],

    var_name="Life Cycle",

    exclude_home=True,   

    save_dir="../Outputs/Charts/",
    save_prefix="Life_Cycle_Trip_Mode_"
)

results_mode = weighted_ttests_full(
    long_df_lifecycle_mode,
    group_var="LIF_CYC_label_abbr",
    category_col="TRIPMODE_label"
)

results_mode_between_capita= results_mode[(results_mode['test_type']=='between_years') & (results_mode['metric']=='trips_per_capita')]
results_mode_within_capita= results_mode[(results_mode['test_type']=='within_year_groups') & (results_mode['metric']=='trips_per_capita')]
results_mode_between_pct= results_mode[(results_mode['test_type']=='between_years') & (results_mode['metric']=='pct_share')]
results_mode_within_pct= results_mode[(results_mode['test_type']=='within_year_groups') & (results_mode['metric']=='pct_share')]

results_mode_between_capita = results_mode_between_capita.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_mode_within_capita = results_mode_within_capita.drop(columns=['test_type', 'metric', 'LIF_CYC_label_abbr', 'year_1', 'year_2'])
results_mode_between_pct = results_mode_between_pct.drop(columns=['test_type', 'metric', 'year', 'group_1', 'group_2'])
results_mode_within_pct = results_mode_within_pct.drop(columns=['test_type', 'metric', 'LIF_CYC_label_abbr', 'year_1', 'year_2'])
results_mode_between_pct = results_mode_between_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_mode_within_pct = results_mode_within_pct.rename(columns={'mean_1': 'pct_1', 'mean_2': 'pct_2'})
results_mode_within_capita = results_mode_within_capita[['group_1', 'group_2', 'TRIPMODE_label', 'year','mean_1', 'mean_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]
results_mode_within_pct = results_mode_within_pct[['group_1', 'group_2', 'TRIPMODE_label', 'year','pct_1', 'pct_2', 'mean_diff', 't_stat','degrees_of_freedom', 'p_value', 'significant','effective_sample_size_1', 'effective_sample_size_2']]

results_mode_between_capita.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripMode_between_years_capita.csv', index=False)
results_mode_within_capita.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripMode_within_year_capita.csv', index=False)
results_mode_between_pct.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripMode_between_years_pct.csv', index=False)
results_mode_within_pct.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripMode_within_years_pct.csv', index=False)

combined_df_lifecycle_mode.to_csv('../Outputs/Tables/LifeCycleGroup_by_TripMode.csv', index=False)


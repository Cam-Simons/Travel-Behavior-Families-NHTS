# Travel-Behavior-Families-NHTS
This repository contains the Python code used for the data cleaning, processing, statistical analysis, and visualization presented in: 

How Household Life-Cycle Stage Shapes Non-Home Travel in the United States, 2001–2022

## Authors
**Cameron Simons, M.S.**
Mineta Transportation Institute
San José State University, San José, CA 95192-0219
ORCID: [0009-0001-2453-4432](https://orcid.org/0009-0001-2453-4432)
Email: cameron.simons@sjsu.edu

**Hilary Nixon, Ph.D.***
Mineta Transportation Institute
San José State University, San José, CA 95192-0219
ORCID: [0000-0001-5378-3473](https://orcid.org/0000-0001-5378-3473)  
Email: hilary.nixon@sjsu.edu

*Corresponding Author

## Overview

This project analyzes data from the National Household Travel Survey (NHTS) to examine travel behavior patterns among different life cycle cohorts within the United States.

The analysis focuses on differences in travel behavior across household life-cycle groups, including changes over time and differences by travel mode and trip purpose.

The code in this repository performs:

- Data cleaning and standardization across NHTS survey years
- Construction of standardized life-cycle group classifications
- Standardized trip mode and trip purpose variables
- Application of NHTS survey weights
- Calculation of weighted descriptive statistics
- Calculation of trip counts and percentage shares
- Statistical comparisons across survey years and life-cycle groups
- Generation of figures and tables used in the report

## Data

The analysis uses data from the following NHTS survey years:

- 2001
- 2009
- 2017
- 2022

The NHTS datasets are publicly available from the National Household Travel Survey program.

The raw NHTS data are **not included in this repository**. Users should obtain the data directly from the appropriate source and place the required files in the directory specified in the analysis scripts.

## Repository Structure
```text
.
├── README.md
├── LICENSE
├── code/
│   ├── NHTS_Life_Cycle_Groups_Core_Metrics.py
│   ├── Trip_Analysis_Time_Series_with_Significance.py
├── outputs/
│   ├── figures/
│   └── tables/
└── requirements.txt

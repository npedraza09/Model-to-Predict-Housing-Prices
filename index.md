# Index

- [Abstract](#Abstract)
- [1. Introduction](#1.-Introduction)
- [2. Extract-Transform-Load](#2.-Extract-Transform-Load)
    - [2.1 The ETL Process](#2.1-The-ETL-Process)
    - [2.2 Data Exploration](#2.2-Data-Exploration)
    - [2.3 Data Preparation](#2.3-Data-Preparation)
    - [2.4 Read the Data Using Python](#2.4-Reading-the-Data-Using-Python)
         - [2.4.1 Reading Sample Data](#2.4.1-Reading-Sample-Data)
         - [2.4.2 Reading the MRST Data](#2.4.2-Reading-the-MRST-Data)
    - [2.5 Writing an Installation Script](#2.5-Writing-an-Installation-Script)
- [3. Analysis and Visualization](#3.-Project-Description)
    - [3.1 Running Queries in MySQL Workbench](#3.1-Running-Queries-in-MySQL-Workbench)
    - [3.2 Running Queries From Python](#3.2-Running-Queries-From-Python)
    - [3.3 Explore Trends](#3.3-Explore-Trends)
    - [3.4 Explore Percentage Change](#3.4-Explore-Percentage-Change)
    - [3.5 Explore Rolling Time Windows](#3.5-Explore-Rolling-Time-Windows)
- [Conclusion](#Conclusion)
- [References](#References)


[Back to top](#Index)

##  Abstract

This project focuses on the analysis of the Monthly Retail Trade Survey (MRTS) data to uncover trends in U.S. retail sales across various business categories. The analysis was conducted through an ETL process, beginning with SQL installation, querying, and extraction of the MRTS data, followed by cleaning and transformation using Python. Key metrics such as total sales, percentage contributions, and rolling averages were computed for categories like food services, sporting goods, and clothing stores. The data visualization revealed key insights, including the rise in spending on sporting goods, the decline of bookstores, and the shrinking gap between men’s and women’s clothing stores in terms of contribution to retail totals. The use of rolling time windows smoothed the data, providing clearer long-term trends. This comprehensive analysis demonstrates how retail sectors have evolved, with notable shifts in consumer behavior over time.


[Back to top](#Index)

## 1. Introduction

In this project, I performed an ETL (Extract, Transform, Load) process and conducted a detailed analysis of the Monthly Retail Trade Survey (MRTS) data. The primary goal was to extract key insights from retail categories, including multiple store categories, and visualize trends over time.

I started by cleaning and formatting my CSV file. I then connected to a MySQL database where the MRTS data was stored. The extraction process involved writing SQL queries to retrieve data for selected retail categories over multiple years. Then, I filtered the data for key categories such as food services, sporting goods, hardware stores, toy stores, men’s and women’s clothing, and bookstores.

After extraction, I transformed the data using Python's pandas library. This step included cleaning the data by handling NULL values, reshaping the data from wide to long format, and calculating important metrics like total sales and percentage contribution of each category. To handle missing values, I used interpolation methods and calculated rolling averages for better trend analysis.

For analysis, I first plotted multiple graphs for trend analysis. I then calculated percentage changes and the contribution of each category to total retail sales. I also explored rolling time windows (4-month and 12-month), which helped smooth out seasonal and short-term fluctuations, providing a clearer view of long-term trends.

Finally, I visualized the data using Matplotlib. Graphs of trends, percentage changes, rolling averages, standard deviations, maximums, and minimums were created for each retail category to illustrate spending trends, volatility, and sales peaks. The visualizations showed key trends, such as the rise in sporting goods sales, the declining share of traditional bookstores, and the narrowing contribution gap between men’s and women’s clothing categories.

This project provided valuable insights into how consumer spending behaviors are changing over time across various retail sectors.


[Back to top](#Index)

## 2. Extract-Transform-Load

[Back to top](#Index)

### 2.1 The ETL Process

#### 1. Extract:  
Identify and research the source of data. Take the raw data from the Excel file.

#### 2. Transform:  
Clean, adjust, and transform the data for better readability and adaptation to an SQL database.

#### 3. Load:  
Insert the clean data to a MySQL database by writing code that allows you to read the data properly and querying over it.


[Back to top](#Index)

### 2.2 Data Exploration

The MRTS Dataset produces the most comprehensive data available on retail economic activity in the United States. These data are widely used throughout government, academic, and business communities. The Bureau of Economic Analysis uses the estimates to calculate Gross Domestic Product. The Bureau of Labor Statistics uses the estimates to develop consumer price indexes and productivity measurements. The Council of Economic Advisers uses the estimates to analyze current economic activity. The Federal Reserve Board uses the estimates to assess recent trends in consumer purchases. The media use the estimates to report news of recent consumer activity. Financial and investment companies use the estimates to measure recent economic trends.

Default fields for each year tab of the original .XLSX file include the NAICS Code, Kind of Business, the month and year each value represents, and a total column for the year. Estimates with a coefficient of variation greater than 30% or with a total quantity response rate less than 50% have been suppressed from publication. These estimates have been replaced with an "S" in the published table. "GAFO" represents stores classified in the following NAICS codes:  442, 443, 448, 451, 452, and 4532. NAICS code 4532 includes office supplies, stationery, and gift stores.) Estimates are adjusted for seasonal variations and holiday and trading-day differences, but not for price changes. Cumulative seasonally adjusted sales estimates are not tabulated.


[Back to top](#Index)

### 2.3 Data Preparation

1. I began by editing the Excel file. I deleted all columns that I found unnecessary and extra info that was not needed in my database. I only took the "Not Adjusted" values, as recommended in one of the previous videos by Dr. Sanchez.
2. I followed this by creating a new .XLSX file called "MRTS_Master", where I joined the data from every year. The columns I kept were: "NAICS Code", "Kind of Business", the months of the year(each month is one column), "Total", and I added a new column called "Year" to represent every year as an index.
3. Once I organized all the data into one file, I transformed this file into a .csv file.


[Back to top](#Index)

### 2.4 Read the Data Using Python

#### Option 1 ####
One way that Python can read a .csv file is by using its CSV package. This package takes each line of the CSV file as a list object. CSV files have a series of delimited values. The most common delimiter is the comma, but in some other scenarios, we can see a semi-colon.

#### Option 2 ####
Another way to read the data from the CSV file is by using the pandas library from Python. This library allows you to read the CSV file and immediately convert it to a data frame.


[Back to top](#Index)

### 2.4.1 Reading Sample Data

I have created a sample dataset called "users". This is a CSV file. This dataset has some normal entries, but other entries have an "(S)", "(NA)", or an empty cell to simulate these scenarios from the MRTS dataset.

In my project, I will be using option 2 to read the data from my sample dataset. 
```python
import pandas as pd # Import Pandas library

sample_df= pd.read_csv('users.csv') # Read CSV file and convert to a dataframe

sample_df # Check the sample dataframe
```

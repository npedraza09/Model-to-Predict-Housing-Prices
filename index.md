# Index

- [Abstract](#Abstract)
- [1. Introduction](#1.-Introduction)
- [2. The Data](#2.-The-Data)
    - [2.1 Import the Data](#2.1-Import-the-Data)
    - [2.2 Data Exploration](#2.2-Data-Exploration)
    - [2.3 Data Preparation](#2.3-Data-Preparation)
    - [2.4 Correlation](#2.4-Correlation)
- [3. Project Description](#3.-Project-Description)
    - [3.1 Linear Regression](#3.1-Linear-Regression)
    - [3.2 Analysis](#3.2-Analysis)
    - [3.3 Results](#3.3-Results)
    - [3.4 Verify Your Model Against Test Data](#3.4-Verify-Your-Model-Against-Test-Data)
- [Conclusion](#Conclusion)
- [References](#References)


[Back to top](#Index)

##  Abstract

In this project, I developed a housing sale price prediction model using linear regression, focusing on feature selection and correlation analysis to identify the most impactful variables. After cleaning the dataset, we used features with strong positive and negative correlations (above 0.4 or below -0.4) to predict housing prices. Missing values were handled by imputing the mean, and the selected features were used to train the linear regression model. The model's performance was evaluated using the R^2 score, which measures how well the model explains the variance in housing prices. Residual analysis, through histograms of prediction errors, showed a reasonably symmetric distribution, indicating accurate predictions with low bias. This model provides a simple yet effective approach to predicting housing prices, offering insights into key drivers of value and laying the groundwork for future improvements and applications in real estate analysis.


[Back to top](#Index)


## 1. Introduction

The objective of this project is to build a prediction model for housing sale prices using linear regression, a common statistical method for understanding relationships between variables. The main goal is to use historical data of house features and prices to predict future sale prices based on key attributes.

We began by loading and exploring the dataset, examining various housing attributes such as square footage, neighborhood, number of rooms, and year of construction. Data cleaning was a crucial first step, where we handled missing values, transformed categorical variables, and removed unnecessary columns. For numeric data, missing values were replaced with the mean to maintain data integrity.

Next, we used correlation analysis to identify features that had the strongest relationships with the target variable, SalePrice. Features with a correlation of 0.4 or higher (positive) or -0.4 or lower (negative) were considered important for the prediction model. This narrowed down the variables to those most relevant to predicting housing prices.

We implemented a linear regression model using the selected variables. The model was trained and tested using available data, and we measured its accuracy using the R^2 score, which indicates how well the model predicts sale prices based on the input variables. Additionally, we analyzed the residuals (prediction errors) to check for bias or inconsistencies in the predictions.

Finally, we evaluated the model’s performance and tested it against a new set of data. Through this process, we aimed to create a reliable and accurate housing price prediction model that could provide valuable insights for real estate investors, analysts, and homeowners.


[Back to top](#Index)

## 2. The Data

[Back to top](#Index)

### 2.1 Import the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as norm
from sklearn import linear_model

data = pd.read_csv('houseSmallData.csv')
```
#### Data Description:
* OverallQual: Rates the overall material and finish of the house.
* GrLivArea: Above grade (ground) living area square feet.
* GarageArea: Size of garage in square feet.
* YearBuilt: Original construction date.
* LotArea: Lot size in square feet.
* Neighborhood: Physical locations within Ames city limits.
* TotalBsmtSF: Total square feet of basement area.
* FullBath: Full bathrooms above grade.
* Bedroom: Bedrooms above grade (does NOT include basement bedrooms).
* YearRemodAdd: Remodel date (same as construction date if no remodeling or additions).
* WoodDeckSF: Wood deck area in square feet.
* 1stFlrSF: First Floor square feet.
* GarageCars: Size of garage in car capacity.
* Fireplaces: Number of fireplaces.
* MasVnrArea: Masonry veneer area in square feet.
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms).


[Back to top](#Index)

### 2.2 Data Exploration

For me, the top features that have the most impact on house prices are:
#### 1. Lot and Property Features:
* LotArea: The total size of the lot in square feet is a strong determinant of property value.
* Neighborhood: The location within city limits strongly influences house prices, as certain neighborhoods are more sought after.
#### 2. House Size and Configuration:
* GrLivArea: The above-ground living area is one of the most critical features for determining home value, as larger homes are typically more expensive.
* TotalBsmtSF: Total square feet of basement area. Larger finished basements often increase a home’s value.
* GarageArea: The size of the garage in square feet also contributes to the home value.
#### 3. Quality and Condition:
* OverallQual: Rates the overall material and finish of the house, a key predictor of house price.
#### 4. Bathrooms and Bedrooms:
* FullBath: Full bathrooms above grade are often considered important by buyers.
* BedroomAbvGr: Number of bedrooms above grade, although after a certain number, this may show diminishing returns.
#### 5. Year Built and Renovation:
* YearBuilt: The original construction date, as newer homes are typically more expensive.
* YearRemodAdd: Remodel date. Homes that have been updated or remodeled are often valued higher.
#### 6. Additional Features:
* WoodDeckSF: The square footage of any wood deck, which increases property value by providing outdoor living space.



#### **Code:**
```python
copy = data.copy() # Copy of data to not affect the original data.
salePrice = copy['SalePrice']

plt.hist(salePrice) # Histogram of the sale price to analyze the price trend.

# Histogram indicates to be similar to a log distribution function. The histogram is skewed, therefore we must analyze the skew:
salePrice.skew()

plt.hist(np.log(salePrice)) # Takes the natural log of the sale price to get a graph that is more similar to a normal distribution.

np.log(SalePrice).skew() # Check the skew again for certainty.

# Our prediction will be made for the log of the sale price, which can also tell us the prediction of just the sale price. We want to make our target prediction the log of the sale price.
target = np.log(salePrice)

plt.scatter(copy['LotArea'], y = target)
# The scatter plot suggests a weak positive correlation between LotArea and SalePrice (after log transformation), but the variability in prices for similar lot sizes indicates that other factors also strongly influence house prices. The presence of outliers also suggests that some large lots do not necessarily command high sale prices.

neigh = pd.DataFrame({'neighborhood': copy['Neighborhood'], 'price': salePrice})
neigh.sort_values('price', ascending = False)[0:6]
# Although this small DataFrame does not help us with our model to predict housing prices, nor with analyzing the correlation between the variable and the prediction, it is helpful to know what the most expensive neighborhoods are for some extra knowledge.

plt.scatter(copy['GrLivArea'], y = target)
# The scatter plot shows a strong positive correlation between GrLivArea and SalePrice (after the log transformation). This means GrLivArea is likely a very important predictor for determining house prices. The relationship appears much stronger compared to the previous variable (LotArea), with fewer outliers and more consistent price trends as the living area increases.

plt.scatter(copy['TotalBsmtSF'], y = target)
# TotalBsmtSF is a fairly strong predictor of sale price, as shown by the positive correlation in this scatter plot. While the relationship is slightly weaker than GrLivArea, it is still a significant factor, especially for homes with larger basements. The presence of some outliers, especially for smaller basements, suggests that basement size alone may not fully determine the sale price, but it is an important contributing factor.

plt.scatter(copy['GarageArea'], y = target)
# The scatter plot suggests that there is a positive relationship between GarageArea and the target variable, but there are also other factors that may influence the target variable because of some of the variability that we notice.

plt.scatter(copy['OverallQual'], y = target)
# There is a clear upward trend, suggesting that higher OverallQual values are associated with higher target variable values. There are a few outliers, but they do not significantly affect the overall trend.


plt.scatter(copy['BedroomAbvGr'], y = target)
# The scatter plot shows a weak positive correlation between BedroomAbvGr and the target variable. This means that as the number of bedrooms above grade (BedroomAbvGr) increases, the target variable tends to increase slightly, but the relationship is not very strong.

plt.scatter(copy['FullBath'], y = target)
# The scatter plot shows a moderate positive correlation between the FullBath and the target variable. This means that as the number of full bathrooms (FullBath) increases, the target variable also tends to increase, but the relationship is not as strong as in some of the previous plots.

plt.scatter(copy['YearBuilt'], y = target)
# YearBuilt shows a slight positive correlation with the Sale Price. It shows a trend for more modern houses increasing the sale price. Although this shows a clear relationship, some other variables such as TotalBsmtSF show to be a stronger predictor.

plt.scatter(copy['YearRemodAdd'], y = target)
# Overall, the scatter plot suggests a positive relationship between YearRemodAdd and the target variable, but it is not a very strong one. Other factors influence stronger the target variable.

plt.scatter(copy['WoodDeckSF'], y = target)
# The scatter plot shows a weak positive correlation between WoodDeckSF and the target variable. The points are clustered in a vertical band on the left side of the plot, indicating that most houses have a small or no deck. There are a few outliers with larger WoodDeckSF values, which might be affecting the correlation. The upward trend is not very pronounced, suggesting that other factors might be more influential on the target variable.
```


[Back to top](#Index)

### 2.3 Data Preparation

#### Search for Null values
```python
sum(copy.isnull().sum() != 0) # Counts how many columns have null values in them.

# Now we want a data frame with the 17 columns with null values. With this information, we can make sure that the independent variables that we use for our prediction are not in this list.
nulls = pd.DataFrame(copy.isnull().sum().sort_values(ascending = False))[0:17]
```
#### Extract numeric values
```python
numeric = copy.select_dtypes(include = [np.number]).interpolate().dropna(axis=1) # Extracts all numeric values from our data, getting rid of columns with null values, and interpolating where there are missing values.
```


[Back to top](#Index)

### 2.4 Correlation

In my data, we are looking for all relevant data that can correlate to the sale price. I will not take into account strings and NaN values.  The correlation will help me make an educated guess for my housing price by indicating me how the independent variables behave with the dependent variable. We will be taking variables that have a positive and negative correlation to the sale price; it is good to know what factors increase with the sale price and which ones increase as the sale price decreases.

To get a better mean accuracy score of my model, I decided that I would use all numeric values that can relevantly correlate with the sale price. After researching, I noticed that a correlation of 0.4 is already considered a variable with a moderate correlation to the desired variable. Therefore, I will take into account |0.4| as a correlation value for my prediction. In my code, I will specify that all variables that have a correlation of 0.4 or higher, or -0.4 or lesser, are collected for a better prediction. I believe that by using as many useful independent variables as I can, I can create a better model to predict housing prices. Although, it is important to set a correlation value boundary, as I did not want to fall under including too many irrelevant or weakly correlated variables which can lead to overfitting. Overfitting happens when the model performs well on training data but poorly on unseen data because it's too complex and captures noise instead of real patterns.

```python
numeric.shape # This outputs (100, 39), therefore there are 39 columns with numeric values.

corrs = numeric.corr() 
spCorrs = corrs['SalePrice'].sort_values(ascending = False) # This outputs all numeric correlations with 'SalePrice'.
spCorrs.drop(['Id','Unnamed: 0']) # I dropped 'Id' and 'Unnamed: 0' as they are not relatable to the sale price.

strongCorrs = spCorrs[(spCorrs >= 0.4) | (spCorrs <= -0.4)].index # Outputs all indexes with variables that have correlations with the sale price of 0.4 or higher, or -0.4 or lesser. 'strongCorrs' has 15 independent variables that will be used for the model prediction.
```
I noticed that most of the variables I chose and analyzed for data exploration at the beginning are within the 15 variables outputted by strongCorrs.



[Back to top](#Index)

## 3. Project Description

In my analysis, I included more variables to correlate than Dr. Williams to get a better prediction for the housing prices. However, the variables that I added have a correlation of 0.4 or more, or 0.4 or less with respect to the sale price. I did not want to fall into making a model that only worked with the data it was trained on, but that is also capable of adapting to new data. 

I noticed that variables related to the area of different parts of the house are correlated to the price as one might think; normally the bigger the house, the more expensive it is. I also noticed that the total number of spaces for something or of something determined the price; again relating to the size or capacity of the house. At last, I saw that the moderately correlated variables were the additional features that a home would have, such as a fireplace, or a wood deck square footage.


[Back to top](#Index)

### 3.1 Linear Regression

Linear Regression roots from linear statistics and is a machine learning algorithm that is used to predict values within a continuous range rather than classifying them into categories. The variables that you want to predict are the dependent variables. The predictions can be made using one or more independent variables. In this project, we use Multiple Linear Regressions. It is essentially an extension of Simple Linear Regression with multiple sets of data used to make the prediction. 

    Multiple Regression Formula: y = β0 + β1x1 + β2x2 + ... + βnxn
    
In this project, I used the Multiple Regression Formula for my algorithm. This formula can be used as an algorithm to predict housing sale prices by establishing a linear relationship between a dependent variable (the target, in this case, the sale price of a house) and multiple independent variables (features like square footage, number of bedrooms, location, etc...).

To implement Linear Regression into Python, we can import libraries such as 'numpy', 'pandas', and 'sklearn'. These libraries allow us to organize the data for further analysis and to do operations on it to get an algorithm that models predictions for housing sale prices.


[Back to top](#Index)

### 3.2 Analysis 

#### Sample 1: 
For the independent variables I am taking into account every column that is a numeric value and that has a correlation of 0.4 or more, or 0.4 or less with respect to the sale price.
```python
x = data[strongCorrs]
y = x['SalePrice']
x = x.drop(['SalePrice'], axis = 1)
x = x.fillna(x.mean()) # Fills NaN values with the mean of x of the respective columns.

from sklearn import linear_model # Library for Linear Regression use in Python
lr = linear_model.LinearRegression()
model = lr.fit(x, y) # Uses the linear regression algorithm to model a prediction for the housing sale prices
predictions = model.predict(x) # Predictions for class labels for samples in x.
```

#### Sample 2: 
For the independent variables I am choosing the top three variables with the most correlation with respect to the sale price.
```python
spCorrs = corrs['SalePrice'].sort_values(ascending = False)
sample2 = spCorrs[0:4].index # Chooses the top three independent variables that correlate the most with sale price.

x = data[strongCorrs]
y = x['SalePrice']
x = x.drop(['SalePrice'], axis = 1)
x = x.fillna(x.mean())

lr = linear_model.LinearRegression()
model = lr.fit(x, y)
predictions = model.predict(x)
```

#### Sample 3: 
For this last sample I chose variables that were the top two, the middle two, and the bottom two variables with correlation with respect to the sale price.
```python
spCorrs = corrs['SalePrice'].sort_values(ascending = False)
sample3 = spCorrs.iloc[[0,1,2,7,8,14,15]].index # Chooses the top two, middle two, and bottom two variables from a descending list of correlations with the sale price.

x = data[strongCorrs]
y = x['SalePrice']
x = x.drop(['SalePrice'], axis = 1)
x = x.fillna(x.mean())

lr = linear_model.LinearRegression()
model = lr.fit(x, y)
predictions = model.predict(x)
```


[Back to top](#Index)

### 3.3 Results

#### Sample 1 Results:
```python
print(f'R^2 is {model.score(x, y)}') # Outputs a score of 0.8822658267161305

plt.hist(y - predictions) # This histogram shows a normal distribution that centers around zero. This indicates that the model is generally unbiased and does not systematically overpredict or underpredict the sale prices. 

plt.scatter(y, predictions) # The scatter plot indicates that there is a strong relationship between the sale price in the data and the predictions given by the model.
```
#### Sample 2 Results:
```python
print(f'R^2 is {model.score(x, y)}') # Outputs a score of 0.789158151143107

plt.hist(y - predictions) # The histogram also shows a normal distribution but with a little bit of more deviation. It does not seem to be fully centered around zero. But, it still indicates a generally decent model.

plt.scatter(y, predictions) # The scatter plot outputs a good relationship between x and y, although not with the same strength as the previous sample.
```
#### Sample 3 Results:
```python
print(f'R^2 is {model.score(x, y)}') # Outputs a score of 0.8223724126168643

plt.hist(y - predictions) # The histogram outputs a similar result to the previous ones. It seems more centered around the zero and with less deviation than sample 2, but sample 1 still seems to be the better fitting one for the model.

plt.scatter(y, predictions) # The scatter plot shows a strong relationship as well. It only indicates a slightly weaker relationship than that shown in sample 1.
```


Sample 1 proved to be the better sample as expected. This sample included the most amount of relevant variables related to the correlation between them and the sale price. It outputted a higher mean accuracy score than the two other samples. The graphs showed a stronger relationship and less deviation.


[Back to top](#Index)

### 3.4 Verify Your Model Against Test Data

```python
test = pd.read_csv('jtest.csv') # Read new test data.
test.head() # Check if the data loaded properly and how it looks like.

x = test[strongCorrs] # Use the same model but now with the test data.
y = x['SalePrice']
x = x.drop(['SalePrice'], axis = 1)
x = x.fillna(x.mean())

predictions = model.predict(x)
print(f'R^2 is {model.score(x, y)}') # Outputs a score of 0.8500705283312364

plt.hist(y - predictions) # Outputs a histogram similar to sample 1.

plt.scatter(y, predictions) # Outputs a scatter plot similar to sample 1.

```

The results show to be in accordance with what I found earlier. The mean accuracy of the model showed a good score with the new test data. If it showed a good score with the training data and now with the test data, it indicates that the model is working well and that it adapts to new data properly. The plots also indicate that the model works well because the histogram plotted a normal distribution centered around zero, and the scatter plot showed a strong relationship between the new sale price and predictions.


[Back to top](#Index)

## Conclusion

1. Data Cleaning is Crucial: In developing a housing sale price prediction model, handling missing values was essential to ensure that the model could be trained without errors. Techniques like replacing missing values with column means (fillna(x.mean())) or removing irrelevant columns significantly improved data quality.

2. Feature Selection Improves Model Accuracy: Selecting relevant features with strong correlations (both positive and negative) to the target variable (SalePrice) helped to focus the model on the most important predictors. Variables with correlations greater than 0.4 or less than -0.4 were particularly useful in constructing the regression model, allowing it to capture key patterns and relationships in the data.

3. Linear Regression Model is Effective for Predictive Analysis: Implementing a linear regression model (LinearRegression from sklearn) provided a straightforward and interpretable method for predicting housing prices. The resulting R^2 score, which measures how well the model fits the data, gave a direct indication of the model's performance and accuracy.

4. Residual Analysis Helps Evaluate Model Performance: The residuals (difference between actual and predicted values) were plotted as a histogram, which allowed us to assess the spread and bias in the predictions. A symmetric histogram centered around zero indicates that the model predictions are reasonably accurate with low bias, while a skewed distribution might point to issues such as underfitting or overfitting.

5. Correlation Helps in Understanding Variable Impact: The .corr() function provided insights into which features have the strongest linear relationships with the sale price. This enabled us to focus the model on variables like overall quality, living area, and other important features, and exclude features with weak or no correlation to housing prices.

6. Handling Multicollinearity: While this model focused on single-variable correlations, it is important in future work to handle multicollinearity, where two or more variables are highly correlated with each other. 

7. Interpretability: Linear regression models are highly interpretable compared to other machine learning methods. Each feature’s coefficient provides a direct indication of how much it influences the sale price. This simplicity makes it easier to understand the model's decision-making process.



[Back to top](#Index
)
## References

- Kiernan, Diane. “Chapter 7: Correlation and Simple Linear Regression.” MILNELibrary. Natural Resources Biometrics, 2014. https://milnepublishing.geneseo.edu/natural-resources-biometrics/chapter/chapter-7-correlation-and-simple-linear-regression/ .

- Kiernan, Diane. “Chapter 8: Multiple Linear Regression.” MILNELibrary. Natural Resources Biometrics, 2014. https://milnepublishing.geneseo.edu/natural-resources-biometrics/chapter/chapter-8-multiple-linear-regression/ .

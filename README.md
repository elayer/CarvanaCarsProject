## Carvana Cars Regression Project - Overview:

* Scraped roughly two thousand pages of car data from Carvana's used car listings. 

* Cleaned and processed the data for preparation for EDA and model building tasks. Tasks ranged from imputing missing values to feature engineering new versions of variables which had high cardinality. I also performed some outlier detections techniques using PyCaret as well as Z-score.

* Within Model Building, I began by dropping insignificant attributes as well as variables exhibiting multicollinearity. After using the OLS method from statsmodels, I moved into applying linear regression techniques such as standard Linear Regression, Elastic Net, and Kernel Ridge. I followed this up by applying more powerful models such as Random Forest, LightGBM, and XGBoost regression.

* Built a Flask API framework to enable potential users to make price estimates based on their desired input values.


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, requests, beautiful soup, matplotlib, seaborn, sklearn, optuna, plotly, scipy,
lightgbm, xgboost, pycaret

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Helpful guide on creating geographical maps using plotly:
https://towardsdatascience.com/geographical-plotting-of-maps-with-plotly-4b5a5c95f02a

* Learning resources to read behind the logic of Kernel Ridge regression:
https://www.sciencedirect.com/topics/computer-science/kernel-ridge-regression
https://towardsdatascience.com/svm-kernels-what-do-they-actually-do-56ce36f4f7b8

## Web Scraping:

Created a web scraper using Requests and Beauitful Soup. Using two separate scrapers to first scrape the individual vehicle links and then scrape those links for vehicle information, here is the information collected from the scrapers:

*   Make
*   Model
*   Trim
*   Year
*   Mileage
*   City
*   State
*   Curb Weight
*   MPG-City, MPG-Highway
*   Engine Cylinder Count
*   Fuel
*   Exterior Color, Interior Color
*   Number of Keys
*   Doors
*   Seating
*   Ac
*   Powered Windows
*   CD Player
*   Powered Locks
*   Tilted Wheel
*   Powered Seats
*   Facets (individual car special features)
*   Imperfections (number of bad points of a car such as damage)
*   Engine Description
*   Transmission
*   Price

## Data Cleaning

After collecting the data, I performed several necessary cleaning and processing tasks to prepare the data for modeling. Below are the steps I followed to clean and transform the data:

* Imputed the missing values from the respective columns with missing data. I used judegment based on the distribution, other attribute values, and the cardinality of columns to decide how to impute each columns' missing values.

* Since the eletric cars did not have any engine cylinders and were the rows where engine cylinder count were missing, I imputed these rows with 0.0 since those cars don't have engine cylinders.

* I then checked columns with links to other attributes that had very high cardinality and dropped them from the analysis.

* With columns that had strong cardinality but common terms, I contorted these features into new categorical attributes with more broad values. These were new columns for Engine Description and Transmission. 

* Using the facets attriubte, I attempted to find common but not universal aspects of cars I thought would drive a car's price given the years of the cars come between around 2008 to 2021. These attributes were Bluetooth and Rear View Camera. I made sure to create too many attributes using this strategy.

* I converted boolean attributes such as power locks to binary integers 0 and 1.

* Next, I label encoded the remaining categorical attributes, since some had moderate cardinality and creating dummy variables would have made model building cumbersome by engendering the curse of dimensionality.

* Highly skewed attributes were log transformed.

* I used PyCaret to detect outliers, which I ultimately elected to leave in the dataset as there were no clear patterns among them when inspecting the rows. Below is an image of the t-SNE graph from PyCaret's anomalt detection.

![alt text](https://github.com/elayer/CarvanaCarsProject/blob/main/outlier_detections.png "LDA Topic Example")

## EDA
As one would expect, Year was generally positively correlated with price, while mileage was negatively correlated with price. Of course, some brands of cars were still more espensive than others even with more miles, meaning that brand/make is indeed an influencer of car price. 

Based on the data collected, most of the states with higher average car prices were in the Northeast of the United States. 

Interestingly enough, the number of imperfections only started to noticeably decrease the car price after 3 imperfections. This may imply you could still get a car priced as high as a car with nothing tarnished as long as there aren't too many of them.

Cars with lower amounts of enginr cylinders had higher gas mileages, yet were priced lower than cars with lower gas milages and more engine cylinders. One could infer more fuel efficient cars are generally cheaper than ones that are not.

![alt text](https://github.com/elayer/CarvanaCarsProject/blob/main/carprice_bybrand.png "Price by Brand")
![alt text](https://github.com/elayer/CarvanaCarsProject/blob/main/mileage_bybrand_price.png "Price by mileage for Brands")
![alt text](https://github.com/elayer/CarvanaCarsProject/blob/main/geomap_prices.png "Average Price per State Map")
![alt text](https://github.com/elayer/CarvanaCarsProject/blob/main/mpgcity_price_bycyl.png "MPG per Engine Cylinder Price")

The Toyota Prius was at the top of fuel effieincy within the data.

## Model Building
Using the statsmodels OLS method, I was able to knock out some tasks simutaneously but checking p-values of variables and dropping variables that were multicollinear.

* Following using the OLS method, I made models using Linear Regression, Kernel Ridge, and Elastic Net to juxtapose models with regularization methods. 

* Following this, I utilized tree-based models such as Random Forest, LightGBM, and XGBoost Regression. 


## Model Performance (Sentiment Classification)
The statsmodels OLS and standard Linear Regression models achieved similar performance metrics. Applying polynomial features to Ridge regression with Kernel Ridge sharply improved model metrics (RMSE and R2). Following this, Random Forest also achieved strong performace, but LightGBM and XGBoost achieved the highest performance metrics of all the attempted models to no surprise. Below are the current best performance metrics per model attempted (on the test sets):

* Statsmodels OLS) R2: 76.30

* Linear Regression) R2: 75.71, RMSE: 0.1564

* Kernel Ridge Regression) R2: 85.32, RMSE: 0.1216

* Elastic Net) R2: 71.68, RMSE: 0.1688

* Random Forest Regression) R2: 85.14, RMSE: 0.1223

* LightGBM Regression) R2: 92.19, RMSE: 0.0887

* XGBoost Regression) R2: 93.44, RMSE: 0.0812

I attempted to use Optuna to optimize the hyperparameters of Kernel Ridge and XGBoost. However, I didn't obtained any better models than the ones with hyperparameter sets I established manually. Attempts to use GridSearchCV did not converge.

## Future Improvements
With this project now having a functional FlaskAPI, I could experiment using different models in production. It currently uses the best Kernel Ridge model. While it may not have the same peak performance as the XGBoost regressor model, we know there will sometimes be a trade-off between performance and interpretability.

I also could explore some stacking methods for modeling, or even create a neural network to maximize performance of car price prediction while still being able to potentially include some high cardinality attributes such as car model.

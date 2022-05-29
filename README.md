## Carvana Cars Regression Project - Overview:

* Scraped roughly two thousand pages of car data from Carvana's used car listings. 

* Cleaned and processed the data for preparation for EDA and model building tasks. Tasks ranged from imputing missing values to feature engineering new versions of variables which had high cardinality. I also performed some outlier detections techniques using PyCaret as well as Z-score.

* Within Model Building, I began by dropping insignificant attributes as well as variables exhibiting multicollinearity. After using the OLS method from statsmodels, I moved into applying linear regression techniques such as standard Linear Regression, Elastic Net, and Kernel Ridge. I followed this up by applying more powerful models such as Random Forest, LightGBM, and XGBoost regression.


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
*   Year
*   Mileage
*   State
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

After collecting data, I performed several necessary text cleaning steps in order to analyze the corpus and perform EDA. I went through the following steps to clean and prepare the data:

* Loaded the spacy English corpus and updated the stop words list to include /n and /t

* With each review separated in a separate list, I lemmatized the text to keep only the root word and lowercased each word

* Then, I only kept words that were not punctuation and were either numeric or alphabetic characters of text

* Lastly, in order to maintain the integrity of the reviews, I dropped reviews that were less than 15 characters long to maintain reviews conducive to NLP algorithms. I also removed reviews more than 512 characters long for the PyTorch model to operate on the reviews correctly

## EDA
Some noteable findings from performing exploratory data analysis can be seen below. I found from looking at the Bi-Grams of the words in the reviews corpus, a lot of them primarily vaunted the game, compared Elden Ring to similar games, or were geared at pointing out performance issues. Similar sentiments can be seen in uni and tri-grams as well (in EDA notebook). The Chi2 influential term analysis graph is the most interesting to me. 

I found words that primarily distinguish between positive and negative reviews dealt with screen ultrawide support and performance issues such as crashes. The last picture looks at the LDA results chart, with one topic being comprised of positive comments about the game itself. The second topic comprises mainly of words related to the performance of the game and frame rate issues. The third topic composes primarily of reviews resenting the game's difficulty.

![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/bigrams_picture_2.png "BiGrams Counts")
![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/chi2_picture.png "Chi2 Influential Words")
![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/lda_picture_4.png "LDA Topic Example")

With the most relevant terms and reviews left over, I think we have found the most prevalent topics in the reviews corpus. Those three being positive elements of the game, resentful reviews aimed at the game's difficulty, and the performance issues the game has. Therefore, forcusing on the areas of improvement, the game could perhaps allow for adjustment of difficulty, as well as work on ameliorating the frame rate and other performance related problems.

## Model Building (Sentiment Classification)
Before building any models, I transformed the text using Tfidf Vectorizer and Count Vectorizer in order to make the data trainable. 

* I started model building with Naive Bayes. From here, confusion matrix results improved as I moved to using the SGD classifier, and then Logistic Regression. 

* I then attempted to use PyTorch with the HuggingFace Transformer library (namely, using RoBERTa) to maximize sentiment classification results. Although RoBERTA with PyTorch performed better than Logistic Regression, Logistic Regression achieved good results as well albeit the recall for non-recommended reviews being low. 


## Model Performance (Sentiment Classification)
The Naive Bayes, SGDClassifier, and Logistic Regression models resptively achieved improved results. I then built the PyTorch model with HuggingFace. Since training the entire model with PyTorch using just 4 Epochs and 5 folds for cross validation would have taken more than 4 days on my computer, I only used on epoch on one fold. After this, I gathered the results of the model based on only that much training.

<b>(The possible labels for classification here are 0 : Non-recommended and 1 : Recommended)</b>

Below are the Macro F1 Scores of each model built:

* Naive Bayes: 0.48

* SGD Classifier (SVM using Hinge Loss): 0.69

* Logistic Regression: 0.81

* RoBERTa with PyTorch: 0.87 (after only 1 epoch on 1 fold of data)

With a more powerful machine, I think we can achieve a robust model knowing the granular differences between recommended and non-recommended reviews. Here is an example of some predictions made from the model using a few samples from another fold that model wasn't trained on:

![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/1foldpreds.png "Example PyTorch Predictions")

## Future Improvements
I came back to remove remove words from the N-gram analysis to locate more genuine phrase occurences. I was able to dig up more relevant review content to the game this way.

It's sometimes difficult to locate all of the insincere reviews, especially on Steam. However, I think this could lead to more elaborate and discrete topics potentially being found.

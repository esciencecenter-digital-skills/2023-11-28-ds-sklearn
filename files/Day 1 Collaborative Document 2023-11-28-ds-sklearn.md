![](https://i.imgur.com/iywjz8s.png)


# Day 1 Collaborative Document 2023-11-28-ds-sklearn

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [https://tinyurl.com/2023-11-28-day-1](https://tinyurl.com/2023-11-28-day-1)

Collaborative Document day 1: [https://tinyurl.com/2023-11-28-day-1](https://tinyurl.com/2023-11-28-day-1)

Collaborative Document day 2: [https://tinyurl.com/2023-11-28-day-2](https://tinyurl.com/2023-11-28-day-2) 

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2023-11-28-ds-sklearn)

🛠 Setup

[link](https://github.com/INRIA/scikit-learn-mooc/blob/main/local-install-instructions.md)


## 👩‍🏫👩‍💻🎓 Instructors

Sven van der Burg, Flavio Hafner, Malte Luken

## 🧑‍🙋 Helpers

Carlos Murilo Romero Rocha

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda day 1
* 09:30	Welcome and icebreakerat
* 09:45	Introduction to machine learning
* 10:30	Break
* 10:40	Tabular data exploration
* 11:30	Break
* 11:40	Fitting a scikit-learn model on numerical data
* 12:30	Lunch Break
* 13:30	Fitting a scikit-learn model on numerical data
* 14:30	Break
* 14:40	Fitting a scikit-learn model on numerical data
* 15:30	Break
* 15:40	Handling categorical data
* 16:15	Wrap-up
* 16:30	END

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises

### Exercise: Machine learning concepts

Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.


#### What kind of problem is it?

a) a supervised problem (x)
b) an unsupervised problem
c) a classification problem
d) a regression problem (x)

Select all answers that apply

#### What are the features?

a) the number of rooms might be a feature (x)
b) the post code of the house might be a feature (x)
c) the price of the house might be a feature

Select all answers that apply

#### What is the target variable?

a) the full text description is the target
b) the price of the house is the target (x)
c) only house description with no price mentioned are the target

Select a single answer

#### What is a sample?

a) each house description is a sample (x)
b) each house price is a sample
c) each kind of description (as the house size) is a sample

Select a single answer


### Exercise: Data exploration (15min,  in groups)

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `../datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:
1. How many features are numerical? How many features are categorical?


3. What are the different penguins species available in the dataset and how many samples of each species are there?



5. Plot histograms for the numerical features
6. Plot features distribution for each class (Hint: use `seaborn.pairplot`).
7. Looking at the distributions you got, how hard do you think it will be to classify the penguins only using "culmen depth" and "culmen length"?
- Ron: quite easy, the three species are almost completely separatable by those two dimensions

#### Solution

```
# Load dataset
penguins = pd.read_csv("datasets/penguin_classification.csv")

# Get overview of features and target
penguins.head()

target_column = "Species"

# Get counts of different species
penguins[target_column].value_counts()

# Make histogram of numerical features
_ = penguis.hist()

# Plot distributions of features for different species
sns.pairplot(penguins, hue="Species")

```

### 📝 Exercise : Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```



#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the data and target loaded above
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.




### Exercise: Compare with simple baselines
#### 1. Compare with simple baseline
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```


### Exercise: Recap fitting a scikit-learn model on numerical data [Sven]
#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data [CORRECT]
c) to improve the accuracy of the model

Select all answers that apply



#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function [CORRECT]
b) calling fit to train the model on the training set and score to compute the score on the test set [CORRECT]
c) calling cross_validate by passing the model, the data and the target [CORRECT]
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply


#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets [CORRECT]
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set [CORRECT]
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets [CORRECT]

Select all answers that apply



## 🧠 Collaborative Notes
### Icebreaker :ice_cream:
Sven asked people to form a map of the Netherlands according to where they work; to gather according to what they ate for breakfast and their knowledge in machine learning.

### Introduction to Machine Learning

Goal: Building predictive models by learning systematic patterns from data

Generalization: Making good predictions for unseen data

Different sources of variance:

- Explained: Variance due to know variables (e.g., age)
- Noise: Variance from unknown sources (e.g., measurement error)

Memorizing: Matching a case to the most similar learned instance

Train data: Data from which the predictive model has learned
Test data: Unseen data on which the predictive models is tested and evaluated (never used for training!)

Data matrix (tabular data):

- Rows are observations, samples, cases, instances
- Columns are features, variables, predictors

Supervised machine learning:

- Data matrix X with n observations
- Target y with n observations
- Goal: predict target y from X
- Example: Linear regression

Unsupervised machine learning:

- Only data matrix X but no target
- Example: Clustering

Regression: Target y is continuous (e.g., test score 1-100)
Classification: Target y is discrete (e.g., test passed/not passed)

### Coding Setup

- Open Anaconda Prompt (Windows) or Terminal (Mac, Windows)
- To activate conda environment:
    - Navigate to folder scikit-learn-mooc with `cd`
    - Activate environment with `conda activate scikit-learn-course`
- To open Jupyter lab: `jupyter lab`
- To open new notebook: Blue '+' sign in top left
- To check that everyting works: Type `import pandas as pd` into first cell and execute with Ctrl + Enter or Shift + Enter
- To turn a code cell into a Markdown cell: Click into cell and press Esc and then M

### Tabular Data Exploration

Make sure to open the notebook in the notebooks folder.

```
import pandas as pd

# Load dataset
adult_census = pd.read_csv("../datasets/adult-census.csv")

# Print first 5 rows of dataset to get an overview
adult_census.head()

# Name of target column we want to predict
target_column = 'class'

# Count different levels of target 
adult_census[target_column].value_counts()

# Names of numerical and categorical columns
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

all_columns = numerical_columns + categorical_columns + [target_column]

# Get row and column numbers
adult_census.shape

# Create histograms of numerical variables
_ = adult_census.hist(figsize = (20, 14))

# Count different levels of categorical variables
adult_census["sex"].value_counts()

adult_census["education"].value_counts()

# Count co-occurrence of two columns
pd.crosstab(
    index=adult_census["education"], columns=adult_census["education-num"]
)

# Make a nice plot of the data
import seaborn as sns

# We plot a subset of the data to keep the plot readable and make the plotting
# faster
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]
_ = sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)

# Create decision rule by hand
_ = sns.scatterplot(
    x="age",
    y="hours-per-week",
    data=adult_census[:n_samples_to_plot],
    hue=target_column,
    alpha=0.5,
)

```

Class imbalance: One class is much more frequent than the other(s)

- Can cause problems for making predictions

### Fitting a scikit-learn Model on Numerical Data
#### Load data

```

import pandas as pd

# Load dataset
adult_census = pd.read_csv("datasets/adult-census-numeric.csv")

# Get overview of features
adult_census.head()

# Separate features and target
target_name = "class"
target = adult_census[target_name]

# Remove target from feature set
data = adult_census.drop(columns=target_name)
data.head()

# Get row and column numbers
data.shape
```

#### Fit model and make predictions

```
from sklearn.neighbors import KNeighborsClassifier

# Create model object from model class
model = KNeighborsClassifier()

# Fit model to features and target
_ = model.fit(data, target)

# Predict target from features
target_predicted = model.predict(data)

# Get first five predicted values
target_predicted[:5]

# Compare with first five target values
target[:5]

# Calc overal accuracy
(target == target_predicted).mean()

```

![](https://codimd.carpentries.org/uploads/upload_f3eafb23d9ce189736048a223789d412.png)

![](https://codimd.carpentries.org/uploads/upload_e12ec2111aaf97a98d2a0e5b6c99a492.png)

#### Evaluate on test data

```python
# load a hold out dataset
adult_census_test = pd.read_csv("datasets/adult-census-numeric-test.csv")

# get the column 'class'
target_name = "class"
target_test = adult_census_test[target_name]
# drop the column 'class'
data_test = adult_census_test.drop(columns=[target_name])

# check the dimension of your data test
data_test.shape

# check accuracy on unseen data
model.score(data_test, target_test)
```

![](https://codimd.carpentries.org/uploads/upload_290b4b7b24855669bf8953a1c8cb0c1f.png)

<u>Different types of methods we can use in scikit-learn:</u>

- `model.fit`: performs the calibration/fit of your model based on the input dataset (features + targets)
- `model.predict`: uses your fitted model and input features to predict/output the predicted targets
- `model.score`: uses your fitted model to predict targets and compares them with real/expected targets. This outputs accuracy, i.e., number of correct predictions / total number of predictions


#### Working with numerical data
```python
# load full dataset
adult_census = pd.read_csv("datasets/adult-census.csv")

# drop/remove "education-num" column
adult_census = adult_census.drop(columns="education-num")

# check first entries of your dataset
adult_census.head()
 
# define features and target directly
(data, target) = adult_census.drop("class"), adult_census["class"]

# check the types of data you have in as features
data.dtypes

# create a list with the name of numerical columns
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

# store all numeric data columns in a single variable
data_numeric = data[numerical_columns]
```

#### Train-test split the dataset
```python
from sklearn.model_selection import train_test_split

# split full data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data_numeric, target, random_state=42, test_size=0.25)

# we will use the logistic regression model
# if `0.1 * age + 3.3 * hours-per-week - 15.1 > 0, predict high-income` otherwise predict low_income

from sklearn.linear_model import LogisticRegression

# define the model
model = LogisticRegression()

# use train data to calibrate the model
model.fit(data_train, target_train)

# predict how good your model is or how well it generalizes
accuracy = model.score(data_test, target_test)
```

#### How to pick a model ?
![](https://codimd.carpentries.org/uploads/upload_21db196b479d16a3a497e2234883ecc5.png)


#### Model fitting with preprocessing
```python
# asks pandas to give an overview of the features
data_train.describe()

from sklearn.prepocessing import StandardScaler

# scale your features
scaler = StandardScaler()
scaler.fit(data_train)
data_train_scaled = scaler.transform(datatrain)

# alternative to fit and transform
data_train_scaled = scaler.fit_transform(data_train)
```

![](https://codimd.carpentries.org/uploads/upload_44c7dae06bf70f41d5c3e2723feb4bad.png)

![](https://codimd.carpentries.org/uploads/upload_b33bab904659b55878ed54d9def2ca28.png)

#### Using pipeline to scale features and fit

```python
from sklearn.pipeline import make_pipeline

# define a pipeline with a scaler and model
model = make_pipeline(StandardScaler(), LogisticRegression())

# scale and fit directly with your pipeline
model.fit(data_train, target_train)

# when predicting and scoring, we also use the pipeline
predicted_target = model.predict(data_test)
model.score(data_test, target_test)
```
![](https://codimd.carpentries.org/uploads/upload_0cdc43a661dedfa526b3f7c845b1043c.png)

![](https://codimd.carpentries.org/uploads/upload_eec40162dfe36a9367e4e56d9dc0b82a.png)

#### Cross-validation

- the score of your model depends on the way you split your train and test sets.
- we could employ methods to split your data differently randomly and them perform various fits with different choices of train-test splits using for instance k-folds.
- Cross-validation evaluates the variability of estimation of the generalization performance.

example:

```python
from sklearn.model_selection import cross_validate

# for the cross-validation pass all data
cv_result = cross_validate(model, data_numeric, target, cv=5)
```

## 💬 Feedback Morning Day 1
Can you give us some feedback about the course so far?
Please write down one thing that went well and one thing that can be improved: 
Think about the pace, the instructors, the content, the interaction, the coffee, location. Any feedback is welcome!

### What went well?
* Everything is very clear
* Clear and easy to follow, good pace
* Everything is very clear, sometimes the shortcuts go to fast
* The workshop is amazing! It is exactly the thing I was looking for, as I have some skills in python, but I didn't know how to do ML.
* very good, just a little bit slow. I think I sat in the worst possible seat (first row in 3.05 from the window) from which you cannot see well in the main board or the 2nd screen. Very helpful that many instructors are in the class, so any problem is resolved fast. very good planning.
* Clear and easy to follow
* Clear and good examples
* Everything is good
* Everything nice
* I enjoyed this first day and it was easy to follow
* Very clear and easy to follow. 
* I liked the workshop so far, I could follow and understand
* Up to now, it is clear and easy to follow and with some background theory information. 
* very clear and well structured
* Very clear. 
* very fun course, clear examples. The walkthrough with the code is very useful 

### What can be improved?

* Maybe entertain somehow variable speeds, but thats not easy I guess
* Not much so far, I'd like to have references and extra materials. 
* data importing and cleaning could maybe be a bit faster, but was also okay with this speed
* I don't have suggestions so far.
* a bit more time to write along
* maybe some theory on what the different models are doing under the hood, their differences etc
* Not so fast, especially when coding

## 💬 Feedback Afternoon Day 1

#### What went well ?
- good pace
- good balance between theory and practice
- clear and easy to follow
- good pace and easy to follow
- clear, applicable and a good structure for guiding the course
- Speed is good and everything is clear
- pace is really good and it's easy to follow due to clear explanations
- understandable
- very good explanations on our questions
- To me seems important contents are getting covered, and I learned new stuff which is nice. I liked the exercies because without it I couldn't get the whole point completely
- the sharing file was quite useful in getting up to speed when distracted
- excellent, and a great atmosphere
- To me seems important contents are getting covered, and I learned new stuff which is nice

#### Points to improve ?
- can't really think of anything to improve
- less breaks more content (but this is indiviudal specific I guess)
- I would like to see many more optional tasks, as there is always substantial slack time between and within tasks
- can't think of anything


## Sven's summary:
* It will become more interesting as we move along
* Raise your hand if you missed a part in the type along

## 📚 Resources

- Installation instructions for the anaconda environment: https://github.com/INRIA/scikit-learn-mooc/blob/main/local-install-instructions.md
- Fairness in machine learning: https://www.fairness.org/
- Coursera Machine Learning Specialization with Andrew Ng: https://www.coursera.org/specializations/machine-learning-introduction
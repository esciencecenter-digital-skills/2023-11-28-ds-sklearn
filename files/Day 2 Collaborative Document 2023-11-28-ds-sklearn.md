![](https://i.imgur.com/iywjz8s.png)


# Day 2 Collaborative Document 2023-11-28-ds-sklearn

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------
This is the Document for today: [https://tinyurl.com/2023-11-28-day-2](https://tinyurl.com/2023-11-28-day-2)

Collaborative Document day 1: [https://tinyurl.com/2023-11-28-day-1](https://tinyurl.com/2023-11-28-day-1)

Collaborative Document day 2: [https://tinyurl.com/2023-11-28-day-2](https://tinyurl.com/2023-11-28-day-2) 


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

### Additional super important guideline:
**Please do not use the coffee machine ☕️**
 
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

## 🗓️ Agenda day 2
09:30	Welcome, look at feedback, recap exercise from yesterday
09:45	Handling categorical data
10:30	Break
10:40   Handling categorical data
11:30	Break
11:40	Handling categorical data
12:30	Lunch Break
13:30   Finish 'Combining numerical and categorical data'
14:00   Intuitions on linear models and tree-based models
14:30   Break
14:40   Overfitting and underfitting
14:55	Validation and learning curves
15:10   Try out learned skills on penguins dataset
15:30   Break
15:40   Try out learned skills on penguins dataset
16:00   Machine learning best practices; Q&A
16:20   Wrap-up & Post-workshop Survey
16:30	Drinks

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

#### Ordinal encoding (5 minutes in pairs, then discussion): [Flavio] 

Q1: Is ordinal encoding is appropriate for marital status? For which (other) categories in the adult census would it be appropriate? Why?


Q2: Can you think of another example of categorical data that is ordinal?


Q3: What problem arises if we use ordinal encoding on a sizing chart with options: XS, S, M, L, XL, XXL? (HINT: explore `ordinal_encoder.categories_`)



Q4: How could you solve this problem? (Look in documentation of OrdinalEncoder)



Q5: Can you think of an ordinally encoded variable that would not have this issue?



#### Exercise: The impact of using integer encoding for with logistic regression (groups of 2, 15min)

Goal: understand the impact of arbitrary integer encoding for categorical variables with linear classification such as logistic regression.

We keep using the `adult_census` data set already loaded in the code before. Recall that `target` contains the variable we want to predict and `data` contains the features.

If you need to re-load the data, you can do it as follows:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```


**Q0 Select columns containing strings**
Use `sklearn.compose.make_column_selector` to automatically select columns containing strings that correspond to categorical features in our dataset.

**Q1 Build a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier**

You'll need the following, already loaded modules:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
```

Because OrdinalEncoder can raise errors if it sees an unknown category at prediction time, you can set the handle_unknown="use_encoded_value" and unknown_value parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for more details regarding these parameters.


**Q2 Evaluate the model with cross-validation.**

You'll need the following, already loaded modules:

```python
from sklearn.model_selection import cross_validate

```

**Q3 Repeat the previous steps using an `OneHotEncoder` instead of an `OrdinalEncoder`**

You'll need the following, already loaded modules:

```python
from sklearn.preprocessing import OneHotEncoder

```


#### Exercise: The impact of feature preprocessing on a pipeline that uses a decision-tree-based classifier [in pairs; 10 min]

Again, load the data first:
```python=
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

Q1: Measure the accuracy of the decision tree classifier (the reference model).

Q2: Now write a similar pipeline that also scales the numerical features using `StandardScaler` (or similar). 
How does this compare to the reference?

Q3: Now let's see if we can improve the model using One-Hot encoding. Recreate the pipeline, but using one-hot encoding instead of ordinal encoding. 
How does this compare to the previous 2 pipelines?
*Note*: `HistGradientBoostingClassifier` does not yet support sparse input data. Use `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use of a dense representation as a workaround.



##### SOLUTIONS

- Q1

```python
categorical_preprocessor = OrdinalEnconder()

preprocessor = ColumnTransformer(
    [("categorical", categorical_preprocessor, categorical_columns)],
    remainder="passthrough"
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)

_ = model.fit(data_train, target_train)
model.score(data_test, data_target)
```

- Q2

```python
preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, categorical_columns),
        ("numerical", numerical_preprocessor, numerical_columns")
    ]
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

_ = model.fit(data_train, target_train)
model.score(data_test, data_target)
```

- Q3

```python
categorical_preprocessor = OneHotEnconder(handle_unknown='ignore', sparce_output=False)

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns")
    ]
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

_ = model.fit(data_train, target_train)
model.score(data_test, data_target)
```

### Bonus Exercise: Try out learned skills on penguins dataset
In this exercise we use the [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/)

We use this dataset in classification setting to predict the penguins’ species from anatomical information.

Each penguin is from one of the three following species: Adelie, Gentoo, and Chinstrap. See the illustration below depicting the three different penguin species:

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/palmer_penguins.png)

Your goal is to predict the species of penguin based on the available features. Start simple and step-by-step expand your approach to create better and better models.

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/culmen_depth.png)

You can load the data as follows:
```python
penguins = pd.read_csv("../datasets/penguins_classification.csv")
```

### Exercise: overfitting and underfitting

#### 1: A model that is underfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply


## 🧠 Collaborative Notes

### Icebreaker :ice_cream:
Sven performed a brief stretching section.


### Handling categorical data
open a new notebook:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

# create your target set
target_name = "class"
target = adult_census[target_name]

# define your features columns
data = adult_census.drop(columns=[target_name])

# print a summary of one categorical feature
data["native-country"].value_counts().sort_index()

# check the data types of your features
data.dtypes
```

#### Select features based on their data type

Here is how to select categorical features directly

```python
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns

# generating a pandas DataFrame with all categorical features
data_categorical = data[categorical_columns]
data_categorical.head()
```

#### Encoding categorical values

- Ordinal encoding: encodes each category with a different number

```python
from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]

encoder = OrdinalEncoder().set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded.head()

# check the order in which sklearn assign the categories
encoder.categories_

# check the encoding applied on all categorical features.
data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()
```

- OneHot encoding: considers nominal variables without an order

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded.head()

data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()
```

##### When to use which kind of encoding?
- linear models: one-hot encoding
- tree-based models: ordinal encoding

#### Evaluate our predictive pipeline
```python
data["native-country"].value_counts()
```
`handle_unkown="ignore"` defines how we deal with categories that are in the test data but not in the training data. From the documentation: When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros. In the inverse transform, an unknown category will be denoted as None.
```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=500)
)

from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data_categorical, target)
```

### Combining categorical and numerical features (Malte)
```python
from sklearn.compose import make_column_selector as selector

# Create a selector for numerical columns
numerical_columns_selector = selector(dtype_exclude=object)

# Create a selector for categorical columns
categorical_columns_selector = selector(dtype_include=object)
```

We forgot this step before, but it is of course necessary to actually select the categorical and numerical columns:
```python
categorical_columns = categorical_columns_selector(data)
numerical_columns = numerical_columns_selector(data)
```

Define data transformers/preprocerssors for different data types:
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
categorical_preprocessor = OneHotEncoder(handle_unkown="ignore")
numerical_preprocessor = StandardScaler()
```


Dispatch the different transformations to the different columns:
```python
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard-scaler", numerical_preprocessor, numerical_columns)
    ])
```

Define a pipeline where we first transform the data using our previously defined column transformer, and then train/predict a Logistic Regression model.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
```
```python
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)
```

Train the full pipeline: (fitting the transformers and the model)
```python
_ = model.fit(data_train, target_train)
```

Make predictions for first 5 cases:
```python
model.predict(data_test)[:5]
```

### Decision tree classifier
```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
```

```python
categorical_preprocessor = OrdinalEncoder(
    handle_unkown="use_encoded_value", unknown_value=-1
)
preproccessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, categorical_columns)
    ],
    remainder="passtrough"
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())
```

```python
_ = model.fit(data_train, target_train)
```

```python
model.score(data_test, target_test)
```

### Intuitions on linear models and tree-based models

#### Linear Models for regression

- Linear regression is very intuitive and easier to use/understand
- Example: Estimating house prices, *i.e.* 
  - Price = a * Living_Area + b * House_Size +...+ c
- Can be used with: 
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
...
```

#### Linear models for classification

- Use logistic regression:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)
...
```

**Note**: linear models show severe limitations for more complicated datasets, as they have low flexibility.

#### Decision Trees

- Used both for regression and classification.
- The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
- <u>For classification</u>: it works by splitting the dataset, stablishing boundaries that help in the process of classification
- <u>For regression</u>: the splitting adds steps, creating a "step" function to describe your data
- the *max_depth* of the tree-model controls the trade-off between underfitting and overfitting
- Mostly useful as a buiding blocks for ensemble models like *Random Forests* and *Gradient Boosting Decision Trees* 


### Overfitting and Underfitting

*Idea*: try to understand whether your model generalizes well, *i.e.*, if it reproduces well unseen data

- Overfitting: model is too complicated to reproduce your data. It may reproduce exactly the data set used for fitting but is not capable of describing new data outside it.
- Underfitting: model is to simple to be able to reproduce complex patterns present in your dataset.

### Validation and learning curves

#### Comparing train and test errors


*Idea*: compare errors on test data (generalization) with the ones on training data as we make the model more complex.

- there is always a "sweet-spot" which shows a nice balance between train and test errors. This determines the optimal flexibility for our model with the lowest generalization error

#### Varying sample size

*Idea*: compare train and test errors as a function of the size of dataset (with constant # of model parameters)

- there will be a point in which both train and test errors converge. This is the point where we do no need to increase the size of our dataset. 

**Notes**
- your <u> model overfits </u> when you have a very small dataset and/or a very complex model
- your <u> model underfits </u> when you have a very large dataset and model with very low flexibility

## 💬 Feedback Morning Day 2
### What went well 💪:
- Clear answers to my questions
- Getting more interesting
- The messages were very clear.
- Speed it good and explanations are clear during coding
- Good pace
- nice speed, clear explanations (x2)
- Clear and concise
- more interesting content, good speed
- Interesting content
-  clear explanations and interesting content

### What can be improved 🙀:
- as it gets more complex, may be nice to have checkpoints like making it more interactove with questions(or tricky questions) etc.
- The question in the exercises could a little bit be more clear
- I would personally like some more background info on how the different models actually work. Not necessarily a full math lecture, but just some diagrams that visualize the different methods so it becomes intuitive when to use which type of model.(x3)
- A bit more background theory
- better separation between practical and explanation; it's hard to follow both at the same time. 
- I have nothing to add here.
- i would be interested in more theory to get more background on why you choose certain parameters. I think it would be nice to start off with more theory and then apply it.
- Wait for everyone before starting. If one thing is missing at the start and everyone continues, it is impossible to keep up and understand. I do not learn from just copying what is done. Check/wait for the audience more frequently. 
- Some more explanation of what's happening behind the functions
- sometimes a line of code is executed and then the output moves the code up and makes it non-visible to us, making it hard to write along
- short slide for each method
- Exercises are not clear sometimes. Not clear what needs to be done in comparison with what we did during demonstrations
- would be good to have things explained before doing actual coding
- Maybe adding more commenting on the collaborative document

## 💬 Feedback Afternoon Day 2

### What went well 💪:
- I enjoyed the theoretical part

- Theoretical part was helpful and interesting
- The theoritical training was very helpul
- I enjoyed the penguin exercise
### What can be improved 🙀:
-Theory before the exercises maybe?
- Theory in the morning and then applying
- I would have loved it if we knew the theory before the practical parts
- hind 
- 
## Sven's summary:
- Great, more things to improve! This means its getting interesting and more complicated :) How can we improve today?:
    - Take a bit more time for this exercise to make sure everyone is on track
    - Help us out by stopping us if something is unclear (we will also try to give more room for it)
- For next time:
    - We will shorten the 'Combination of numerical and categorical features' part, leave out the part on decision trees (because we haven't introduced it yet) and leave some more time for questions
    - We will improve the exercise descriptions for this part

maybe explain the concept first then go to the practice

## 📚 Resources

[Post-workshop survey](https://www.surveymonkey.com/r/PL2YK9J)
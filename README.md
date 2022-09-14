# Football-Results-classification

<img src="/images/se1113l151-serie-a-logo-lega-serie-a-european-leagues.png" width="100" >

## Goal

*Championship used Serie A Year 2021/2022*

The aim of this project is to try to classify if a game is going to end with a number of goal greater than 2.5,we have the odds of the bookmaker BET365 that can be converted into probabilities and we can use those odds as referencee point ,but our purpose would be beating those odds.

The strategy that i want to adopt is based on goals scored and goals conceded:

I will compute the average goals scored and conceded for each team from all the previous matches

and the average goals scored and conceded for each team from the the last n previous matches (that it can be 1 or 2 and so on..)

The ideal goal would be to find differences between the performance of teams between the last n previous games and total games useful to classify the target

## Library used
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.figure_factory as ff
import seaborn as sns
```

## Loading Dataset

> Firstly, i scraped  information about the matches from flashscore website but since that  requires a lot of time ,i performed furthermore reasearch and i founded a website that provides a lots of detailed info,
so i preferred download a ready dataset

https://www.football-data.co.uk/

```
data.shape
(380,105)
```
## EDA and Feature Engineering

In this part of the analysis i will execute the following steps:

>First of all i will create the target variable:
>
>1 -> Over 2.5
>
>0 -> Under 2.5

>After i will compute the accuracy according to the odds of the bet365 website in order to have a metric to compare

>At this point i will perform feature engineering,i should compute the average stats of the last 5(or another number) matches for each team,and the the average stats of the total previous matches for each team

>After that i will check if there are Nan values

## *Target Variables*

> To create the target variables i just need to look at the final result,more precisily i am going to look at the variables *FTHG* and *FTAG*,i will sum their value and convert it into a binary variable 

> 1 means Over 2.5
>
> 0 means Under 2.5

![](https://github.com/datascientist-hist/Football-Results-classification/blob/main/images/count_targettrain.png)

## *Metrics*

**True**

Looking at the frequencies of the target variable we can state that during the championship related to year 2021/2022 we would have a 56% empirical probability of guessing the Over 2.5 outcome by choosing at random

**Prediction by bookmaker**

Instead using the odds offered by bookmaker B365 to choose the event Over 2.5 If we had played according to the odds we would have guessed the 58% of the bets

To be competitive i should do better than those metrics

![](https://github.com/datascientist-hist/Football-Results-classification/blob/main/images/confusionmatrixb365%20total.png)
```
#Precision score over 2.5 and under 2.5
precision_positive = metrics.precision_score(target, predB365, pos_label=1)
precision_negative = metrics.precision_score(target, predB365, pos_label=0)
print(f"Precision score over2.5:  {precision_positive}\nPrecision score under2.5:  {precision_negative}\nAccuracy:  {accuracy}")

Precision score over2.5:  0.5860
Precision score under2.5:  0.5512
Accuracy:  0.5789

```
## *Feature Engineering*

```
info_dict={
'navgGoalHome':navgGoalHome,                 #Feature that describes the average goal scored by home team in the last n matches
'navgGoalConcededHome':navgGoalConcededHome, #Feature that describes the average goal conceded by home team in the last n matches
'navgGoalAway':navgGoalAway,                 #Feature that describes the average goal scored by away team in the last n matches
'navgGoalConcededAway':navgGoalConcededAway, #Feature that describes the average goal scored by away team in the last n matches
'avgGoalHome':avgGoalHome,                   #Feature that describes the average goal scored by home team in the previous matches
'avgGoalConcededHome':avgGoalConcededHome,   #Feature that describes the average goal conceded by home team in the previous matche
'avgGoalAway':avgGoalAway,                   #Feature that describes the average goal scored by away team in the previous matche
'avgGoalConcededAway':avgGoalConcededAway,   #Feature that describes the average goal conceded by away team in the previous matche
'avgshothome':navgShotHome,                  #Feature that describes the average shot on target  by home team in the last n matches
'navgShotAway':navgShotAway                  #Feature that describes the average shot on target  by away team in the last n matches
}

```
Then I added those feature to a new dataframe adding also *Time* feature

## *Verifying Nan values*

I founded only 1 Nan value in *B365>2.5* feature  for this reason i decided to delete the observations instead of imputing the value

## *Distributions of the features*
To analyze the features created before i will look at the empirical distributions using as kernel a gaussian distribution
Below there are two graphs as example:

![](/images/newplot.png)
![](/images/newplot(1).png)

The graphs do not reveal useful information ,I also computed correlation matrix to try to understand better data
![](/images/corr.png)

From the Correlation Matrix we can observe some info like:

avgGoalHome and navgGoalHome have a positive correlation index
avgGoalConcededHome and navgGoalHome have a negative correlation index and so on...
Those information are obvious because navg and avg are generated from the same variables,so also these information are not useful,

## Training Models
At this point we can try to fit some model ,but from the analysis done before i don't guess that the model will able to return a good output or at least i don't guess that i will be able to do better than the random choice

Models that  i am going to use :

 - GaussianNB
 
 - KNeighborsClassifier

 - LogisticRegression

 - RandomForestClassifier
 
 - SVC

 - XGBoost
 - 
 I will use K-fold Cross Validation to test the accuracy of the model

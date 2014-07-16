Practical Machine Learning Assignment 
========================================================

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.


## Data 
The training data for this project is available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data is available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


## Purpose
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. This report describes how we built our model, how we used cross validation, what we think the expected out of sample error will be, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases and validate those in Part II of the assignment. 


## Loading the data

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#Read the training and test data set
personaldata <- read.csv("pml-training.csv")
personaldatatesting <- read.csv("pml-testing.csv")
```


## Data Pre-Processing
A basic step that was taken is to show which of the 159 variables 
countained a large number of NA values. 67 variables contained almost
all NA values. These variables where excluded from the model.


```r
navariables <- character()

for (Var in names(personaldata)) {
    missing <- sum(is.na(personaldata[,Var]))
    if (missing > 0) {
        print(c(Var,missing))
        navariables <- c(navariables,Var)
    }
}
```

```
## [1] "max_roll_belt" "19216"        
## [1] "max_picth_belt" "19216"         
## [1] "min_roll_belt" "19216"        
## [1] "min_pitch_belt" "19216"         
## [1] "amplitude_roll_belt" "19216"              
## [1] "amplitude_pitch_belt" "19216"               
## [1] "var_total_accel_belt" "19216"               
## [1] "avg_roll_belt" "19216"        
## [1] "stddev_roll_belt" "19216"           
## [1] "var_roll_belt" "19216"        
## [1] "avg_pitch_belt" "19216"         
## [1] "stddev_pitch_belt" "19216"            
## [1] "var_pitch_belt" "19216"         
## [1] "avg_yaw_belt" "19216"       
## [1] "stddev_yaw_belt" "19216"          
## [1] "var_yaw_belt" "19216"       
## [1] "var_accel_arm" "19216"        
## [1] "avg_roll_arm" "19216"       
## [1] "stddev_roll_arm" "19216"          
## [1] "var_roll_arm" "19216"       
## [1] "avg_pitch_arm" "19216"        
## [1] "stddev_pitch_arm" "19216"           
## [1] "var_pitch_arm" "19216"        
## [1] "avg_yaw_arm" "19216"      
## [1] "stddev_yaw_arm" "19216"         
## [1] "var_yaw_arm" "19216"      
## [1] "max_roll_arm" "19216"       
## [1] "max_picth_arm" "19216"        
## [1] "max_yaw_arm" "19216"      
## [1] "min_roll_arm" "19216"       
## [1] "min_pitch_arm" "19216"        
## [1] "min_yaw_arm" "19216"      
## [1] "amplitude_roll_arm" "19216"             
## [1] "amplitude_pitch_arm" "19216"              
## [1] "amplitude_yaw_arm" "19216"            
## [1] "max_roll_dumbbell" "19216"            
## [1] "max_picth_dumbbell" "19216"             
## [1] "min_roll_dumbbell" "19216"            
## [1] "min_pitch_dumbbell" "19216"             
## [1] "amplitude_roll_dumbbell" "19216"                  
## [1] "amplitude_pitch_dumbbell" "19216"                   
## [1] "var_accel_dumbbell" "19216"             
## [1] "avg_roll_dumbbell" "19216"            
## [1] "stddev_roll_dumbbell" "19216"               
## [1] "var_roll_dumbbell" "19216"            
## [1] "avg_pitch_dumbbell" "19216"             
## [1] "stddev_pitch_dumbbell" "19216"                
## [1] "var_pitch_dumbbell" "19216"             
## [1] "avg_yaw_dumbbell" "19216"           
## [1] "stddev_yaw_dumbbell" "19216"              
## [1] "var_yaw_dumbbell" "19216"           
## [1] "max_roll_forearm" "19216"           
## [1] "max_picth_forearm" "19216"            
## [1] "min_roll_forearm" "19216"           
## [1] "min_pitch_forearm" "19216"            
## [1] "amplitude_roll_forearm" "19216"                 
## [1] "amplitude_pitch_forearm" "19216"                  
## [1] "var_accel_forearm" "19216"            
## [1] "avg_roll_forearm" "19216"           
## [1] "stddev_roll_forearm" "19216"              
## [1] "var_roll_forearm" "19216"           
## [1] "avg_pitch_forearm" "19216"            
## [1] "stddev_pitch_forearm" "19216"               
## [1] "var_pitch_forearm" "19216"            
## [1] "avg_yaw_forearm" "19216"          
## [1] "stddev_yaw_forearm" "19216"             
## [1] "var_yaw_forearm" "19216"
```

```r
#Number of variables with almost all NA's
length(navariables)
```

```
## [1] 67
```

```r
#Drop columns that countain mostly NA's
personaldata <- personaldata[,!(names(personaldata) %in% navariables)]
```


The second thing that was done was to remove zero covariate variables meaning variables
that have little variability and therefore would not contribute to our prediction. 

We also removed 5 variables that were part of the tombstone of each obeservations. 



```r
nsv <- nearZeroVar(personaldata ,saveMetrics=TRUE)
nsv
```

```
##                         freqRatio percentUnique zeroVar   nzv
## X                           1.000     100.00000   FALSE FALSE
## user_name                   1.101       0.03058   FALSE FALSE
## raw_timestamp_part_1        1.000       4.26562   FALSE FALSE
## raw_timestamp_part_2        1.000      85.53155   FALSE FALSE
## cvtd_timestamp              1.001       0.10193   FALSE FALSE
## new_window                 47.330       0.01019   FALSE  TRUE
## num_window                  1.000       4.37264   FALSE FALSE
## roll_belt                   1.102       6.77811   FALSE FALSE
## pitch_belt                  1.036       9.37723   FALSE FALSE
## yaw_belt                    1.058       9.97350   FALSE FALSE
## total_accel_belt            1.063       0.14779   FALSE FALSE
## kurtosis_roll_belt       1921.600       2.02324   FALSE  TRUE
## kurtosis_picth_belt       600.500       1.61553   FALSE  TRUE
## kurtosis_yaw_belt          47.330       0.01019   FALSE  TRUE
## skewness_roll_belt       2135.111       2.01305   FALSE  TRUE
## skewness_roll_belt.1      600.500       1.72256   FALSE  TRUE
## skewness_yaw_belt          47.330       0.01019   FALSE  TRUE
## max_yaw_belt              640.533       0.34655   FALSE  TRUE
## min_yaw_belt              640.533       0.34655   FALSE  TRUE
## amplitude_yaw_belt         50.042       0.02039   FALSE  TRUE
## gyros_belt_x                1.059       0.71348   FALSE FALSE
## gyros_belt_y                1.144       0.35165   FALSE FALSE
## gyros_belt_z                1.066       0.86128   FALSE FALSE
## accel_belt_x                1.055       0.83580   FALSE FALSE
## accel_belt_y                1.114       0.72877   FALSE FALSE
## accel_belt_z                1.079       1.52380   FALSE FALSE
## magnet_belt_x               1.090       1.66650   FALSE FALSE
## magnet_belt_y               1.100       1.51870   FALSE FALSE
## magnet_belt_z               1.006       2.32902   FALSE FALSE
## roll_arm                   52.338      13.52563   FALSE FALSE
## pitch_arm                  87.256      15.73234   FALSE FALSE
## yaw_arm                    33.029      14.65702   FALSE FALSE
## total_accel_arm             1.025       0.33636   FALSE FALSE
## gyros_arm_x                 1.016       3.27693   FALSE FALSE
## gyros_arm_y                 1.454       1.91622   FALSE FALSE
## gyros_arm_z                 1.111       1.26389   FALSE FALSE
## accel_arm_x                 1.017       3.95984   FALSE FALSE
## accel_arm_y                 1.140       2.73672   FALSE FALSE
## accel_arm_z                 1.128       4.03629   FALSE FALSE
## magnet_arm_x                1.000       6.82397   FALSE FALSE
## magnet_arm_y                1.057       4.44399   FALSE FALSE
## magnet_arm_z                1.036       6.44685   FALSE FALSE
## kurtosis_roll_arm         246.359       1.68179   FALSE  TRUE
## kurtosis_picth_arm        240.200       1.67159   FALSE  TRUE
## kurtosis_yaw_arm         1746.909       2.01305   FALSE  TRUE
## skewness_roll_arm         249.558       1.68688   FALSE  TRUE
## skewness_pitch_arm        240.200       1.67159   FALSE  TRUE
## skewness_yaw_arm         1746.909       2.01305   FALSE  TRUE
## roll_dumbbell               1.022      83.78351   FALSE FALSE
## pitch_dumbbell              2.277      81.22516   FALSE FALSE
## yaw_dumbbell                1.132      83.14137   FALSE FALSE
## kurtosis_roll_dumbbell   3843.200       2.02834   FALSE  TRUE
## kurtosis_picth_dumbbell  9608.000       2.04362   FALSE  TRUE
## kurtosis_yaw_dumbbell      47.330       0.01019   FALSE  TRUE
## skewness_roll_dumbbell   4804.000       2.04362   FALSE  TRUE
## skewness_pitch_dumbbell  9608.000       2.04872   FALSE  TRUE
## skewness_yaw_dumbbell      47.330       0.01019   FALSE  TRUE
## max_yaw_dumbbell          960.800       0.37203   FALSE  TRUE
## min_yaw_dumbbell          960.800       0.37203   FALSE  TRUE
## amplitude_yaw_dumbbell     47.920       0.01529   FALSE  TRUE
## total_accel_dumbbell        1.073       0.21914   FALSE FALSE
## gyros_dumbbell_x            1.003       1.22821   FALSE FALSE
## gyros_dumbbell_y            1.265       1.41678   FALSE FALSE
## gyros_dumbbell_z            1.060       1.04984   FALSE FALSE
## accel_dumbbell_x            1.018       2.16594   FALSE FALSE
## accel_dumbbell_y            1.053       2.37489   FALSE FALSE
## accel_dumbbell_z            1.133       2.08949   FALSE FALSE
## magnet_dumbbell_x           1.098       5.74865   FALSE FALSE
## magnet_dumbbell_y           1.198       4.30129   FALSE FALSE
## magnet_dumbbell_z           1.021       3.44511   FALSE FALSE
## roll_forearm               11.589      11.08959   FALSE FALSE
## pitch_forearm              65.983      14.85577   FALSE FALSE
## yaw_forearm                15.323      10.14677   FALSE FALSE
## kurtosis_roll_forearm     228.762       1.64102   FALSE  TRUE
## kurtosis_picth_forearm    226.071       1.64611   FALSE  TRUE
## kurtosis_yaw_forearm       47.330       0.01019   FALSE  TRUE
## skewness_roll_forearm     231.518       1.64611   FALSE  TRUE
## skewness_pitch_forearm    226.071       1.62573   FALSE  TRUE
## skewness_yaw_forearm       47.330       0.01019   FALSE  TRUE
## max_yaw_forearm           228.762       0.22933   FALSE  TRUE
## min_yaw_forearm           228.762       0.22933   FALSE  TRUE
## amplitude_yaw_forearm      59.677       0.01529   FALSE  TRUE
## total_accel_forearm         1.129       0.35674   FALSE FALSE
## gyros_forearm_x             1.059       1.51870   FALSE FALSE
## gyros_forearm_y             1.037       3.77637   FALSE FALSE
## gyros_forearm_z             1.123       1.56457   FALSE FALSE
## accel_forearm_x             1.126       4.04648   FALSE FALSE
## accel_forearm_y             1.059       5.11161   FALSE FALSE
## accel_forearm_z             1.006       2.95587   FALSE FALSE
## magnet_forearm_x            1.012       7.76679   FALSE FALSE
## magnet_forearm_y            1.247       9.54031   FALSE FALSE
## magnet_forearm_z            1.000       8.57711   FALSE FALSE
## classe                      1.470       0.02548   FALSE FALSE
```

```r
#Drop columns that are Near Zero Var
nzvcolumns<- subset(nsv, nsv$nzv,select=-c(freqRatio,percentUnique,zeroVar,nzv)) 
personaldata <- personaldata[,!(names(personaldata) %in% row.names(nzvcolumns)) ]

#Removing columns that are contain tombstone information
personaldata<- subset(personaldata,select=-c(X,user_name,raw_timestamp_part_1,user_name,raw_timestamp_part_2,cvtd_timestamp,num_window )) 
```


## Modeling
We are now left with 53 variables in our model from our original 159. 
We have decided all these remaining variables into our model. 
We have decided to use the Random Forest Machine learning Algorithm with cross validation with a K fold = 3.


```r
set.seed(2133)
modfit <- train(classe ~., data=personaldata, method="rf",trControl = trainControl(method = "cv",number=3))
```


### In-Sample Error

```r
#This provides an overview of our trained model which has an overall Accuracy of > 98%.
# Ths is based on having this simulation many times.

modfit
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 13082, 13081, 13081 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.001        0.002   
##   30    1         1      8e-04        0.001   
##   50    1         1      0.002        0.003   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Out-of-Sample Error 
We would expect the Out-of-Sample error rate to be higher. The reason for using kfold cross-validation is to get a good idea of what that Out-of-Sample error rate would most likely be. It would somewhat be unsual if our accuracy was below 85%. We would expect something closer to 90%. 
However testing on only 20 cases can lead to unexpected results. 


### Results of our Prediction

```r
modpred <- predict(modfit, personaldatatesting)
#This is what we will be submitting for Part II of this assignment.
modpred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```







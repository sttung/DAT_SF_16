# Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import ShuffleSplit

from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d
output_notebook()

get_ipython().magic(u'matplotlib inline')



# Import the data
data = pd.read_csv('train.csv')
# Check data types and determine what variables have null values.
data.info()
# There are missing values in the age, cabin, and embarked variables. 


# Print out a few rows of the data to confirm it's been imported correctly and to understand the data better.
data.head()


# DATA CLEANING 

# AGE
# Determine what the range and mean of the Age feature are
print data.Age.describe()

# Given that our median is less than our mean, we have a few large outliers (older passengers) and a positively skewed dataset.



# List frequencies for each of the age values
# This will give us a better sense of how we want to approach imputing values. 
print data.Age.value_counts()



# Manually assigning the values would be difficult for 177 missing observations, so we'll use NumPy to randomly assign them.
# We assume that our data follows a normal distribution and have NumPy fill in values that will generally preserve the mean of 
# 29.7 and standard deviation of 14.5. Thereby keeping the dataset's frequency distribution roughly the same. 
data.Age.fillna(np.random.normal(29.7,14.5),inplace=True)



# Check to see how close the new mean and standard deviation are to the original.
data.Age.describe()




# SEX
# Rather than text, we'd like to have a binary numeric value for the sexes. 
# Create new variable 'Female' in place of sex
data['Female'] = data.Sex.map({'male': 0, 'female': 1})
# Print out counts for comparison
print data['Female'].value_counts()

# Check that the value counts for 'Female' (new var) match 'Sex' counts
# Print out counts for the original 'Sex' variable
print data['Sex'].value_counts()



# EMBARKED
# Replace string values with numeric values
data.Embarked.replace(['S', 'C', 'Q'],[1,2,3],inplace=True)
# Check counts
data.Embarked.value_counts()
# Filter out 2 observations where we're missing data for embarked
data2 = data[data.Embarked.notnull()]



# FEATURE NORMALIZATION
# Given that our features are not on the same scale (max of 80 for age, max of 512 for fare), we need to normalize our features.
# Create an updated dataframe with features to normalize - drop string features, since those cannot be normalized.
# Drop Sex feature because we've created the Female feature in its place.
data_tn = data2.drop(['Name','Ticket','Cabin','Sex'], axis=1)
# print data_tn.describe()

# Import standardscaler for normalizing mean to 0
from sklearn.preprocessing import StandardScaler

# # Normalize the data
fit_data = StandardScaler().fit_transform(data_tn)
data_n = pd.DataFrame(fit_data, columns=data_tn.columns)
data_n.head()

# # Replace standardized Survived with our previous Survived column. We'll be using this as the target.
data_n ['Survived'] = data['Survived']
data_n.head()
data_n.info()



# CONSTRUCT FEATURES AND TARGET

# Create final features dataframe
# (1) Drop PassengerID feature - unlikely this is a predictor of anything. Rather, it's just an ordered ID. 
# (2) Drop Survived feature - this is our target. 
# (3) We dropped the string features (Name, Ticket, Cabin) earlier - we continue to exclude them. 
# Justification for each: name - if there was predictive value, it'd likely be tied into class - which we 
# can measure through pclass already. Ticket: we do not have information on how these were issued, unlikely that they would 
# provide us with sufficient predictive power to include them. Cabin - we have over 600 missing values and it'd be difficult
# to meaningfully impute values for cabin assignments. Embarked - currently excluded. 
features = data_n.drop(['PassengerId', 'Survived'], axis=1)

# Specify target
target = data_n['Survived']



# LOGISTIC REGRESSION - Model 1

# Import cross_validation function
from sklearn import cross_validation, datasets, svm

# Build our model on features and target specified above.
model_lr = LogisticRegression(C=1)
model_lr.fit(features, target)

cross_validation.cross_val_score(model_lr,features,target,cv=5).mean()



# Define logistic regression model to calculate coefficients later
model_lr1 = LogisticRegression(C=1).fit(features, target)

# Set x's range to be the number of features
x = np.arange(len(features.columns))
# Set names for each column to be the feature names
names = features.columns
names



# Plot feature importance, as measured by coefficients on each of the features
p = figure(title="Model Coefficients")

for val in x:
    p.quad(top=model_lr1.coef_.ravel()[val], 
           bottom=0, left=val+0.2,right=val+0.8, 
           color=['red','orange','yellow', 'green', 'blue', 'purple', 'brown'][val],
           legend=names[val])
# Set the range for y based on minimum and max values of the feature coefficients
p.y_range = Range1d(min(model_lr1.coef_.ravel())-0.1, max(model_lr1.coef_.ravel())+0.5)
show(p)



# We want to see the actual numeric coefficients for each of the features
# We zip together the feature names and corresponding values
coeffs = pd.DataFrame(zip(features,model_lr1.coef_.ravel()),columns=['features','coeff'])
# Create the absolute values of coefficients
coeffs['abs'] = np.absolute(coeffs.coeff.values)
# Sort the coefficients by value 
# coeffs.sort('abs',ascending=False)



# LOGISTIC REGRESSION - MODEL 2

# SETTING UP THE DATA AGAIN
# FEATURE NORMALIZATION
# We drop embarked this time. 
data_tn2 = data.drop(['Name','Ticket','Cabin','Sex','Embarked', 'Fare', 'Parch'], axis=1)
data_tn2.info()

# Import standardscaler for normalizing mean to 0
# from sklearn.preprocessing import StandardScaler

# Normalize the data
fit_data2 = StandardScaler().fit_transform(data_tn2)
data_n2 = pd.DataFrame(fit_data2, columns=data_tn2.columns)

# Replace standardized Survived with our original Survived column. We'll be using this as the target.
data_n2['Survived'] = data['Survived']
data_n2.info()



# Create feature dataframe 
features2 = data_n2.drop(['PassengerId', 'Survived'], axis=1)
# print features2

# Specify target
target2 = data_n2['Survived']



# Import cross_validation function
from sklearn import cross_validation, datasets, svm

# Build our model on features and target specified above.
model_lr = LogisticRegression(C=1)

cross_validation.cross_val_score(model_lr,features2,target2,cv=5).mean()



# Define logistic regression model to calculate coefficients later
model_lr2 = LogisticRegression(C=1).fit(features2, target2)


# To test out new models:
model_lr2.predict()

# Set x's range to be the number of features
x2 = np.arange(len(features2.columns))
# Set names for each column to be the feature names
names = features2.columns
names



# Plot feature importance, as measured by coefficients on each of the features
p = figure(title="Model Coefficients")

for val in x2:
    p.quad(top=model_lr2.coef_.ravel()[val], 
           bottom=0, left=val+0.2,right=val+0.8, 
           color=['red','orange','yellow', 'green', 'blue', 'purple', 'brown'][val],
           legend=names[val])
# Set the range for y based on minimum and max values of the feature coefficients
p.y_range = Range1d(min(model_lr2.coef_.ravel())-0.1, max(model_lr2.coef_.ravel())+0.5)
show(p)



# We zip together the feature names and corresponding values
coeffs2 = pd.DataFrame(zip(features2,model_lr2.coef_.ravel()),columns=['features','coeff'])
# Create the absolute values of coefficients
coeffs2['abs'] = np.absolute(coeffs2.coeff.values)
# # Sort the coefficients by value 
coeffs2.sort('abs',ascending=False)



# PLOTTING THE ROC, DETERMINING THE AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
    
    p = figure(title='Receiver Operating Characteristic')
    # Plot ROC curve
    p.line(x=fpr,y=tpr,legend='ROC curve (area = %0.3f)' % roc_auc)
    p.x_range=Range1d(0,1)
    p.y_range=Range1d(0,1)
    p.xaxis.axis_label='False Positive Rate or (1 - Specificity)'
    p.yaxis.axis_label='True Positive Rate or (Sensitivity)'
    p.legend.orientation = "bottom_right"
    show(p)

# Split the training data again into training subset + test subset
from sklearn.cross_validation import train_test_split
train_feat, test_feat, train_target, test_target = train_test_split(features2,target2, train_size=0.4)

# Defining our model again
model_lr3 = LogisticRegression(C=1).fit(train_feat, train_target)



# Calculate our predicted probabilities - define the predicted probabilities variable.
target_predicted_proba = model_lr3.predict_proba(test_feat)



# Print the target predicted probabilities - check they're within the range of 0,1.
target_predicted_proba



# Plot the ROC curve
plot_roc_curve(test_target, target_predicted_proba)

# Finding a good threshold to yield a higher TPR, lower FPR
fpr, tpr, thresholds = roc_curve(test_target, target_predicted_proba[:, 1])
ideal_thresholds = pd.DataFrame(zip(thresholds, tpr, fpr), columns=['threshold', 'tpr', 'fpr'])
ideal_thresholds.head(200)


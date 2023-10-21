#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# In this project, you must apply EDA processes for the development of predictive models. Handling outliers, domain knowledge and feature engineering will be challenges.
# 
# Also, this project aims to improve your ability to implement algorithms for Multi-Class Classification. Thus, you will have the opportunity to implement many algorithms commonly used for Multi-Class Classification problems.
# 
# Before diving into the project, please take a look at the determines and tasks.

# # Determines

# The 2012 US Army Anthropometric Survey (ANSUR II) was executed by the Natick Soldier Research, Development and Engineering Center (NSRDEC) from October 2010 to April 2012 and is comprised of personnel representing the total US Army force to include the US Army Active Duty, Reserves, and National Guard. In addition to the anthropometric and demographic data described below, the ANSUR II database also consists of 3D whole body, foot, and head scans of Soldier participants. These 3D data are not publicly available out of respect for the privacy of ANSUR II participants. The data from this survey are used for a wide range of equipment design, sizing, and tariffing applications within the military and has many potential commercial, industrial, and academic applications.
# 
# The ANSUR II working databases contain 93 anthropometric measurements which were directly measured, and 15 demographic/administrative variables explained below. The ANSUR II Male working database contains a total sample of 4,082 subjects. The ANSUR II Female working database contains a total sample of 1,986 subjects.
# 
# 
# DATA DICT:
# https://data.world/datamil/ansur-ii-data-dictionary/workspace/file?filename=ANSUR+II+Databases+Overview.pdf
# 
# ---
# 
# To achieve high prediction success, you must understand the data well and develop different approaches that can affect the dependent variable.
# 
# Firstly, try to understand the dataset column by column using pandas module. Do research within the scope of domain (body scales, and race characteristics) knowledge on the internet to get to know the data set in the fastest way.
# 
# You will implement ***Logistic Regression, Support Vector Machine, XGBoost, Random Forest*** algorithms. Also, evaluate the success of your models with appropriate performance metrics.
# 
# At the end of the project, choose the most successful model and try to enhance the scores with ***SMOTE*** make it ready to deploy. Furthermore, use ***SHAP*** to explain how the best model you choose works.

# # Tasks

# #### 1. Exploratory Data Analysis (EDA)
# - Import Libraries, Load Dataset, Exploring Data
# 
#     *i. Import Libraries*
#     
#     *ii. Ingest Data *
#     
#     *iii. Explore Data*
#     
#     *iv. Outlier Detection*
#     
#     *v.  Drop unnecessary features*
# 
# #### 2. Data Preprocessing
# - Scale (if needed)
# - Separete the data frame for evaluation purposes
# 
# #### 3. Multi-class Classification
# - Import libraries
# - Implement SVM Classifer
# - Implement Decision Tree Classifier
# - Implement Random Forest Classifer
# - Implement XGBoost Classifer
# - Compare The Models
# 
# 

# # EDA
# - Drop unnecessary colums
# - Drop DODRace class if value count below 500 (we assume that our data model can't learn if it is below 500)

# ## Import Libraries
# Besides Numpy and Pandas, you need to import the necessary modules for data visualization, data preprocessing, Model building and tuning.
# 
# *Note: Check out the course materials.*

# In[203]:


#  the most important libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ## Ingest Data from links below and make a dataframe
# - Soldiers Male : https://query.data.world/s/h3pbhckz5ck4rc7qmt2wlknlnn7esr
# - Soldiers Female : https://query.data.world/s/sq27zz4hawg32yfxksqwijxmpwmynq

# In[204]:


# concat the two datasets in one dataframe 
df_male = pd.read_csv('C:/Users/HTC/Downloads/ANSUR II MALE Public.csv',encoding='latin-1')
df_female= pd.read_csv('C:/Users/HTC/Downloads/ANSUR II FEMALE Public.csv')
new_df = pd.concat([df_female,df_male])
new_df


# ## Explore Data

# In[205]:


#we need physical feature to clasify race for example : color skin , haire texture and face structure 
new_df.info()


# In[206]:


new_df.head()  


# In[207]:


new_df.tail()


# In[208]:


new_df.nunique()


# In[209]:


# check for null values 


# In[210]:


new_df.isnull().sum()


# In[211]:


#feature selection for variables
# our target is DODRace: 1 = White, 2 = Black, 3 = Hispanic, 4 = Asian,
#5 = Native American, 6 = Pacific Islander, 8 = Other
new_df.drop(columns=["Age", "Date", "subjectid", "SubjectId" ,"WritingPreference", "Weightlbs", "Branch","Component","Ethnicity","Installation", "SubjectNumericRace", "PrimaryMOS", "Heightin"
], inplace=True)
new_df


# In[212]:


new_df.nunique()


# In[213]:


print(new_df.Gender  .unique())


# In[214]:


new_df.isnull().sum()


# In[215]:


# there is some outlier we build a model with and without outliers 


# In[216]:


new_df.describe()


# In[217]:


# check for each numerical feature boxplot and skew also we have now two categorical variables
cat_cols=new_df.select_dtypes(include=['object']).columns
num_cols = new_df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)


for col in num_cols:
    print(col)
    print('Skew :', round(new_df[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    new_df[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=new_df[col])
    plt.show()


# In[218]:



# we plot the cacategorical variables using bar plot 


# In[219]:


fig, axes = plt.subplots(2, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')
sns.countplot(ax = axes[0, 0], x = 'Gender', data = new_df, color = 'blue', 
              order = new_df['Gender'].value_counts().index);
sns.countplot(ax = axes[0, 1], x = 'SubjectsBirthLocation', data = new_df, color = 'blue', 
              order = new_df['SubjectsBirthLocation'].value_counts().index);



# In[220]:


# check numeric variable correlation
correlation_matrix = new_df.corr()
correlation_matrix

#acromialheight and axillaheight  have 0.987452 


# In[221]:


import pandas as pd
from scipy.stats import f_oneway

# Perform ANOVA test for 'Category1' and 'NumericalVariable'
anova_result1 = f_oneway(*[group['DODRace'] for name, group in new_df.groupby('Gender')])

# Perform ANOVA test for 'Category2' and 'NumericalVariable'
anova_result2 = f_oneway(*[group['DODRace'] for name, group in new_df.groupby('SubjectsBirthLocation')])

print("ANOVA Result for Gender and DODRace:")
print(anova_result1)
print("\nANOVA Result for SubjectsBirthLocation and DODRace:")
print(anova_result2)


# In[222]:


# we have inbanalce data and outlier we need to deal with we do the preformane with outlier and without same aboruch with unbalanced

ax = sns.countplot(x='DODRace', data=new_df)
ax.bar_label(ax.containers[0]);


# In[223]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming new_df is your DataFrame containing numerical columns

# Select numerical columns
numerical_cols = new_df.select_dtypes(include=["float64", "int64"])

# Create a 3x3 grid of box plots
fig = plt.figure(figsize=(10, 10), dpi=200)

for i, col in enumerate(range(9)):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=numerical_cols.iloc[:, col])
    plt.xlabel(numerical_cols.columns[col])

plt.tight_layout()
plt.show()


# In[224]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming new_df is your DataFrame

# Set up the subplot grid
plt.figure(figsize=(20, 20))
for index, feature in enumerate(new_df.columns[:9]):  # Iterate over the first 9 columns
    if feature != "DODRace":
        plt.subplot(3, 3, index + 1)  # 3x3 grid, index starts from 0
        sns.boxplot(x='DODRace', y=feature, data=new_df)
        plt.xlabel("DODRace")
        plt.ylabel(feature)
        plt.title(f"Boxplot of {feature} by DODRace")

plt.tight_layout()  # Adjust spacing between plots
plt.show()


# In[225]:


# now we deal with categorical variable width encoding


# In[226]:


# we keep the outlier before and after and now we encode the categorecl coulmn 
# we have unbalanced between male and female 

print(new_df['Gender'].unique()) 
new_df['Gender'].value_counts() 


# In[227]:


new_df["Gender"] =new_df["Gender"].map({"Female":0,"Male":1})
new_df["Gender"]


# In[228]:


new_df.isnull().sum()


# In[229]:


print(new_df.Gender  .unique())
new_df['Gender'].value_counts() 


# In[230]:


# now encode SubjectsBirthLocation

print(new_df['SubjectsBirthLocation'].unique()) 


# In[231]:


new_df.SubjectsBirthLocation.value_counts()


# In[232]:


from sklearn.preprocessing import LabelEncoder
# Bad because some will have a great value for the machine learning and we can't priorty it 
countries = [
    'Germany', 'California', 'Texas', 'District of Columbia', 'New Mexico', 'American Samoa', 'Virginia',
    'South Korea', 'Massachusetts', 'Michigan', 'Dominican Republic', 'Colorado', 'United States', 'South Dakota',
    'Louisiana', 'Ohio', 'South Carolina', 'Mississippi', 'Illinois', 'West Virginia', 'New York', 'Iowa',
    'Florida', 'Poland', 'Oklahoma', 'Pennsylvania', 'North Carolina', 'Alabama', 'Wisconsin', 'Arizona',
    'Washington', 'Kentucky', 'Tennessee', 'Connecticut', 'Iceland', 'Kansas', 'Burma', 'Indiana', 'Georgia',
    'Oregon', 'Delaware', 'Jamaica', 'Puerto Rico', 'Mexico', 'Philippines', 'Maryland', 'Hawaii', 'Ukraine',
    'Montana', 'Italy', 'North Dakota', 'Argentina', 'Saint Lucia', 'New Jersey', 'Dominica', 'Peru', 'Israel',
    'Utah', 'Turkey', 'Morocco', 'Nevada', 'Honduras', 'Russia', 'United Kingdom', 'Missouri', 'Serbia',
    'Belgium', 'Minnesota', 'Ecuador', 'Canada', 'Thailand', 'Idaho', 'Trinidad and Tobago', 'Bolivia', 'Wyoming',
    'Panama', 'Nebraska', 'Liberia', 'Kenya', 'Ghana', 'Vietnam', 'China', 'Maine', 'Guyana', 'Haiti', 'Cameroon',
    'New Hampshire', 'Zambia', 'US Virgin Islands', 'Colombia', 'Arkansas', 'Japan', 'Paraguay', 'Chile', 'India',
    'Bulgaria', 'Antigua and Barbuda', 'Korea', 'Alaska', 'Palau', 'Sri Lanka', 'Barbados', 'Rhode Island',
    'Vermont', 'Bangladesh', 'South Africa', 'Nicaragua', 'Grenada', 'Guam', 'Azerbaijan', 'Sudan', 'Venezuela',
    'Fiji', 'Northern Mariana Islands', 'Iran', 'Bosnia and Herzegovina', 'Bermuda', 'Denmark', 'El Salvador',
    'Romania', 'Netherlands', 'Taiwan', 'British Virgin Islands', 'Sierra Leone', 'Cuba', 'Nigeria', 'Costa Rica',
    'Bahamas', 'Portugal', 'France', 'Belize', 'Guadalupe', 'Nepal', 'Senegal', 'Brazil', 'Cape Verde', 'Syria',
    'Singapore', 'Micronesia', 'French Guiana', 'Iraq', 'Ethiopia', 'Egypt', 'Togo', 'Cambodia', 'Lebanon',
    'Ivory Coast', 'Laos', 'Belarus', 'New Zealand', 'South America', 'Guatemala'
]
label_encoder = LabelEncoder()
encoded_countries = label_encoder.fit_transform(countries)
print(encoded_countries)


# In[233]:


new_df['SubjectsBirthLocation'].value_counts()


# In[234]:



us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
    'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
    'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
    'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'American Samoa',
    'District of Columbia', 'Guam', 'Northern Mariana Islands', 'Puerto Rico', 'US Virgin Islands'
]


new_df['new'] = new_df['SubjectsBirthLocation'].apply(lambda x: 1 if any(state.lower() in x.lower() for state in us_states) else 0)

print('''
U.S.  : {}
Other : {}
'''.format(new_df[new_df['new'] == 1].value_counts().sum(),new_df[new_df['new'] == 0].value_counts().sum()))


# In[235]:


new_df['SubjectsBirthLocation'] = new_df['SubjectsBirthLocation'].apply(lambda x: 1 if any(state.lower() in x.lower() for state in us_states) else 0)


# In[236]:


new_df['SubjectsBirthLocation'].value_counts()


# In[237]:


import plotly.express as px

us_count = new_df[new_df['SubjectsBirthLocation'] == 1].shape[0]
other_count = new_df[new_df['SubjectsBirthLocation'] == 0].shape[0]

fig = px.bar(x=['U.S.', 'Other'], y=[us_count, other_count], title='Distribution of "new" Column',color=['U.S.', 'Other'])
fig.update_xaxes(title='Category')
fig.update_yaxes(title='Count')

fig.show()


# In[238]:


new_df[['SubjectsBirthLocation','new']]


# In[239]:


new_df.drop(columns=['new'], inplace=True)


# In[240]:


new_df.info()


# In[ ]:





# In[241]:


new_df["DODRace"].value_counts().plot(kind="pie", autopct='%1.1f%%',figsize=(10,10));


# In[242]:


ax = sns.countplot(x='SubjectsBirthLocation', data=new_df)
ax.bar_label(ax.containers[0]);


# In[243]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create the countplot
ax = sns.countplot(x='SubjectsBirthLocation', data=new_df, hue='DODRace')

# Set plot labels and title
plt.xlabel('SubjectsBirthLocation ')
plt.ylabel('Count')
plt.title('Count of DODRace by SubjectsBirthLocation')

# Show the plot
plt.show()



    


# In[244]:


# Calculating the outliers for each classes 
unique_classes = new_df['DODRace'].unique()
outlier_counts_by_class = pd.DataFrame(columns=['Class', 'Column', 'Outlier Count'])
Q1 = new_df.quantile(0.25)
Q3 = new_df.quantile(0.75)
IQR = Q3 - Q1

# Loop through each class 
for class_label in unique_classes:
    # Filter for class name 
    class_data = new_df[new_df['DODRace'] == class_label]

    # Calculate outlier counts for each column within this class
    for column in class_data.columns:
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        column_outliers = ((class_data[column] < lower_bound) | (class_data[column] > upper_bound))
        outlier_count = column_outliers.sum()

        # Store the outlier count 
        outlier_counts_by_class = outlier_counts_by_class.append({
            'Class': class_label,
            'Column': column,
            'Outlier Count': outlier_count
        }, ignore_index=True)

outlier_counts_by_class.groupby('Column')['Outlier Count'].sum().sum()


# In[245]:


outlier_counts_by_class.groupby('Class')['Outlier Count'].sum()


# In[246]:


drop_DODRace = new_df.DODRace.value_counts()[new_df.DODRace.value_counts() <= 500].index
drop_DODRace


# In[247]:


for i in drop_DODRace:
    drop_index = new_df[new_df['DODRace'] == i].index
    new_df.drop(index = drop_index, inplace=True)

new_df.reset_index(drop=True, inplace=True)
new_df


# In[248]:


new_df['DODRace'].head()


# In[ ]:





# In[249]:


new_df


# In[250]:


new_df.DODRace.value_counts()


# In[251]:


new_df["DODRace"].value_counts().plot(kind="pie", autopct='%1.1f%%',figsize=(10,10));


# In[252]:


# need to remove outlier , encode the counry diffrent delet one coulmn have strone corrlation 


# # DATA Preprocessing
# - In this step we divide our data to X(Features) and y(Target) then ,
# - To train and evaluation purposes we create train and test sets,
# - Lastly, scale our data if features not in same scale. Why?

# In[ ]:





# In[253]:


from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[254]:


# now we split dataset 
X = new_df.drop(columns=["DODRace"])
y = new_df.DODRace
print('''
Shape of X is  : {}
Shape of Y is  : {}
Shape of df is : {}'''.format(X.shape,y.shape,new_df.shape))


# In[255]:


# we split our dataset 
X_train ,X_test, y_train , y_test = train_test_split(X,y,test_size=0.20 , random_state=42,stratify =y)
print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


# In[ ]:





# In[ ]:





# # Modelling
# - Fit the model with train dataset
# - Get predict from vanilla model on both train and test sets to examine if there is over/underfitting   
# - Apply GridseachCV for both hyperparemeter tuning and sanity test of our model.
# - Use hyperparameters that you find from gridsearch and make final prediction and evaluate the result according to chosen metric.

# ## 1. Logistic model

# ### Vanilla Logistic Model

# In[256]:


from sklearn.pipeline import Pipeline

operations = [("scaler", StandardScaler()), ("logistic", LogisticRegression())]

pipeline_logstic = Pipeline(steps=operations)

pipeline_logstic.fit(X_train, y_train)


# In[257]:


# predict on train and test to see the results
y_pred =pipeline_logstic.predict(X_test)
y_train_pred = pipeline_logstic.predict(X_train)


# In[258]:



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_train, y_train, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred)}''')


# In[259]:


# we can see that recall and f1-score in class 3 not preforming well 


# In[260]:


from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score,  f1_score


f1_S = make_scorer(f1_score, average = "weighted")
precision_S = make_scorer(precision_score, average = "weighted")
recall_S = make_scorer(recall_score, average = "weighted")


scoring = {"f1r":f1_S,
           "precision":precision_S,
           "recall":recall_S}


scores = cross_validate(pipeline_logstic, X_train, y_train, scoring = scoring, cv = 10, return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1,11))
df_scores.mean()


# In[261]:


df_scores


# ### Logistic Model GridsearchCV

# In[262]:


operations = [("scaler", StandardScaler()), ("logistic_model", LogisticRegression(max_iter=5000))]
GridModel_logstic = Pipeline(steps=operations)


# In[263]:


param_grid = { "logistic_model__class_weight" : ["balanced", None],
               'logistic_model__penalty': ["l1","l2"],
               'logistic_model__solver' : ['saga','lbfgs','liblinear'],
               'logistic_model__C' :[0.001,0.01, 0.1, 1, 5, 10, 15, 20, 25]
             }
f1_Hispanic =  make_scorer(f1_score, average=None, labels=[3])# Class 3 represent the Hispanic which is the worst scoring for our model and we need to foucs on it 


# In[264]:


grid_Logistick_pipe = GridSearchCV(GridModel_logstic, param_grid = param_grid,scoring=f1_Hispanic, cv=5, return_train_score=True)
grid_Logistick_pipe.fit(X_train,y_train) 


# In[265]:


grid_Logistick_pipe.best_params_


# In[266]:


y_pred_grid =grid_Logistick_pipe.predict(X_test)
y_train_pred_grid = grid_Logistick_pipe.predict(X_train)


# In[267]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(grid_Logistick_pipe, X_train, y_train, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(grid_Logistick_pipe, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred_grid)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_grid)}''')


# In[268]:


from sklearn.metrics import roc_curve, auc
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
plot_multiclass_roc(grid_Logistick_pipe, X_test, y_test, n_classes=3, figsize=(16, 10));


# In[ ]:





# ## 2. SVC

# ### Vanilla SVC model

# In[74]:


from sklearn.svm import SVC
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score


# In[75]:


operations_SVM = [("scaler", StandardScaler()), ("SVC", SVC(probability=True))]
SVM_pipe = Pipeline(steps=operations_SVM)


# In[76]:


SVM_pipe.fit(X_train,y_train)


# In[77]:


y_pred_svm = SVM_pipe.predict(X_test)
y_train_pred_SVM = SVM_pipe.predict(X_train)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(SVM_pipe, X_train, y_train, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(SVM_pipe, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred_SVM)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_svm)}''')


# In[78]:


from sklearn.metrics import precision_score, recall_score,  f1_score 
from sklearn.metrics import make_scorer

########################333###########################delelte delet 
f1_S = make_scorer(f1_score, average = "weighted")
precision_S = make_scorer(precision_score, average = "weighted")
recall_S = make_scorer(recall_score, average = "weighted")


scoring = {"f1r":f1_S,
           "precision":precision_S,
           "recall":recall_S}
################################################################33



scores = cross_validate(SVM_pipe, X_train, y_train, scoring = scoring, cv = 10, return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1,11))
df_scores.mean()
scores = cross_validate(SVM_pipe,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv = 5,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]


# In[79]:


df_scores


# ###  SVC Model GridsearchCV

# In[ ]:





# In[80]:


param_grid = {
    'SVC__C': [0.5, 1],
    'SVC__gamma': ["scale", "auto", 0.01],
    'SVC__kernel': ['rbf', 'linear']
}


# In[81]:


recall_Hispanic = make_scorer(recall_score, average=None, labels=[3])


# In[82]:


operations = [("scaler", StandardScaler()), ("SVC", SVC(class_weight="balanced", random_state=42))]
svc_grid = Pipeline(steps=operations)


# In[83]:


svm_model_grid = GridSearchCV(svc_grid, param_grid, scoring=recall_Hispanic, cv=10, n_jobs=-1, return_train_score=True)


# In[84]:


svm_model_grid.fit(X_train, y_train)


# In[85]:


best_estimator = svm_model_grid.best_estimator_
print(best_estimator)

grid_search_results = pd.DataFrame(svm_model_grid.cv_results_).loc[svm_model_grid.best_index_, ["mean_test_score", "mean_train_score"]]
grid_search_results


# In[86]:


from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Assuming pipe_model is your trained SVM model, and X_train, y_train, X_test, y_test are your data splits
y_pred_svm = svm_model_grid.predict(X_test)
y_train_pred_SVM = svm_model_grid.predict(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(svm_model_grid, X_train, y_train, ax=axes[0], cmap=plt.cm.Blues)
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(svm_model_grid, X_test, y_test, ax=axes[1], cmap=plt.cm.Blues)
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred_SVM)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_svm)}''')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 3. RF

# ### Vanilla RF Model

# In[269]:


from sklearn.ensemble import RandomForestClassifier
operations = [("scaler", StandardScaler()),
              ("RF_model", RandomForestClassifier(random_state=42))]

pipe_modelRF = Pipeline(steps=operations)

pipe_modelRF.fit(X_train, y_train)


# In[270]:


y_pred_RF = pipe_modelRF.predict(X_test)
y_train_pred_RF = pipe_modelRF.predict(X_train)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(pipe_modelRF, X_train, y_train, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(pipe_modelRF, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred_RF)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_RF)}''')


# In[271]:


scores = cross_validate(pipe_modelRF, X_train, y_train, scoring = scoring, cv = 10, return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1,11))
df_scores.mean()
scores = cross_validate(pipe_modelRF,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv = 5,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]


# In[272]:


df_scores


# ### RF Model GridsearchCV

# In[273]:


param_grid = {'n_estimators':[64, 128, 200],
             'max_features':[2, 4, 'sqrt'],
             'max_depth':[2, 3, 4],
             'min_samples_split':[2, 3,4],
             'min_samples_leaf': [2,3,4],
             'max_samples':[0.8, 1]}


# In[274]:


model = RandomForestClassifier(random_state=101, class_weight='balanced')
rf_grid_model = GridSearchCV(model,
                             param_grid,
                             scoring="recall",
                             n_jobs = -1,
                             verbose=2).fit(X_train, y_train)


# In[275]:


best_estimator = rf_grid_model.best_estimator_
print(best_estimator)


# In[276]:


y_pred_rf = rf_grid_model.predict(X_test)
y_train_pred_rf = rf_grid_model.predict(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(rf_grid_model, X_train, y_train, ax=axes[0], cmap=plt.cm.Blues)
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(rf_grid_model, X_test, y_test, ax=axes[1], cmap=plt.cm.Blues)
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train, y_train_pred_rf)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_rf)}''')


# ## 4. XGBoost

# In[277]:


new_df['DODRace'].head(50)


# ### Vanilla XGBoost Model

# In[278]:


from xgboost import XGBClassifier


# In[281]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



# Assuming X_train, y_train, X_test, y_test are your training and testing data

# Create a pipeline with preprocessing and XGBoost Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Optional: StandardScaler for feature scaling
    ('xgb', XGBClassifier())
])


y_train_xgb = y_train.map({1: 0, 2: 1, 3: 2})
y_test_xgb = y_test.map({1: 0, 2: 1, 3: 2})
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train_xgb)


# In[307]:


y_pred_XGB = pipeline.predict(X_test)
y_train_pred_XGB = pipeline.predict(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(pipeline, X_train, y_train_xgb, ax=axes[0], cmap=plt.cm.Blues)
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test_xgb, ax=axes[1], cmap=plt.cm.Blues)
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train_xgb, y_train_pred_XGB)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test_xgb, y_pred_XGB)}''')


# In[289]:


model = XGBClassifier(random_state=42)

scores = cross_validate(model,
                        X_train,
                        y_train_xgb,
                        scoring=['accuracy',
                                 'precision',
                                 'recall',
                                 'f1',
                                 'roc_auc'],
                        cv = 10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# ### XGBoost Model GridsearchCV

# In[290]:


param_grid = {"n_estimators":[50, 100, 200],
              'max_depth':[3,4,5],
              "learning_rate": [0.1, 0.2],
              "subsample":[0.5, 0.8, 1],
              "colsample_bytree":[0.5,0.7, 1]}


# In[291]:


xgb_model = XGBClassifier(random_state=42)


# In[292]:


xgb_grid = GridSearchCV(xgb_model,
                        param_grid,
                        scoring="f1",
                        verbose=2,
                        n_jobs=-1,
                        return_train_score=True)

xgb_grid.fit(X_train, y_train_xgb)


# ---
# ---

# In[293]:


xgb_grid.best_params_


# In[308]:


y_pred_XGB = xgb_grid.predict(X_test)
y_train_pred_XGB = xgb_grid.predict(X_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(xgb_grid, X_train, y_train_xgb, ax=axes[0], cmap=plt.cm.Blues)
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(xgb_grid, X_test, y_test_xgb, ax=axes[1], cmap=plt.cm.Blues)
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train_xgb, y_train_pred_XGB)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test_xgb, y_pred_XGB)}''')


# ---
# ---

# # SMOTE
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

# ##  Smote implement

# In[209]:


get_ipython().system('pip install imblearn')


# In[306]:


from imblearn.over_sampling import SMOTE





# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Create and train SVM model
operations_SVM_SMOT = [("scaler", StandardScaler()), ("SVC", SVC(C=0.5, class_weight='balanced', kernel='linear',
                     random_state=42))]
SVM_pipe_SMOT = Pipeline(steps=operations_SVM_SMOT)


SVM_pipe_SMOT.fit(X_train_smote,y_train_smote)

# Print classification report
y_pred_svm = SVM_pipe_SMOT.predict(X_test)
y_train_pred_SVM = SVM_pipe.predict(X_train_smote)


cm_train = confusion_matrix(y_train_smote, y_train_pred_SVM)
cm_test = confusion_matrix(y_test, y_pred_svm)

# Plot confusion matrices using seaborn
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Training Set')
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Test Set')
plt.show()
print(f'''


------------------Train Results-----------------------------
{classification_report(y_train_smote, y_train_pred_SVM)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred_svm)}''')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Logistic Regression Over/ Under Sampling

# In[317]:


from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report





# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

operations_LGSM = [
    ("scaler", StandardScaler()), 
    ("logistic", LogisticRegression(C=0.1, class_weight='balanced', penalty='l2', solver='liblinear'))
]

pipeline_logstic = Pipeline(steps=operations_LGSM)

pipeline_logstic.fit(X_train_smote, y_train_smote)


y_pred =pipeline_logstic.predict(X_test)
y_train_pred = pipeline_logstic.predict(X_train_smote)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_train_smote, y_train_smote, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train_smote, y_train_pred)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred)}''')


# In[318]:


from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Apply Random Under-sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

operations_LGSM_1 = [("scaler", StandardScaler()), ("logistic", LogisticRegression(C=0.1, class_weight='balanced', penalty='l2', solver='liblinear'))]

pipeline_logstic = Pipeline(steps=operations_LGSM_1)

pipeline_logstic.fit(X_train_rus, y_train_rus)



y_pred =pipeline_logstic.predict(X_test)
y_train_pred = pipeline_logstic.predict(X_train_rus)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

disp_train = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_train_rus, y_train_rus, ax=axes[0])
disp_train.ax_.set_title('Training Set')

disp_test = ConfusionMatrixDisplay.from_estimator(pipeline_logstic, X_test, y_test, ax=axes[1])
disp_test.ax_.set_title('Test Set')

plt.show()

print(f'''
-----------------------------Train Results-----------------------------
{classification_report(y_train_rus, y_train_pred)}
                        
-----------------------------Test Results-----------------------------
{classification_report(y_test, y_pred)}''')



# ## Other Evaluation Metrics for Multiclass Classification

# - Evaluation metrics
# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd

# In[ ]:


from sklearn.metrics import matthews_corrcoef
get_ipython().run_line_magic('pinfo', 'matthews_corrcoef')
matthews_corrcoef(y_test, y_pred)


# In[ ]:


from sklearn.metrics import cohen_kappa_score
get_ipython().run_line_magic('pinfo', 'cohen_kappa_score')
cohen_kappa_score(y_test, y_pred)


# # Before the Deployment
# - Choose the model that works best based on your chosen metric
# - For final step, fit the best model with whole dataset to get better performance.
# - And your model ready to deploy, dump your model and scaler.

# In[ ]:





# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

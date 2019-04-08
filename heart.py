import mysql.connector
import pandas as pd
import seaborn as sns



mydb = mysql.connector.connect(
host="localhost",
user="chandra",
passwd="chandra12v",
db='project')


sql_select_Query = "select * from heart"
cur = mydb.cursor()
cur.execute(sql_select_Query)
records = (cur.fetchall())

heart=pd.DataFrame(records)

heart.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

heart['sex'][heart['sex'] == 0] = 'female'
heart['sex'][heart['sex'] == 1] = 'male'

heart['chest_pain_type'][heart['chest_pain_type'] == 0] = 'very low'
heart['chest_pain_type'][heart['chest_pain_type'] == 1] = 'low'
heart['chest_pain_type'][heart['chest_pain_type'] == 2] = 'high'
heart['chest_pain_type'][heart['chest_pain_type'] == 3] = 'very high'

heart['fasting_blood_sugar'][heart['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
heart['fasting_blood_sugar'][heart['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

heart['rest_ecg'][heart['rest_ecg'] == 0] = 'very low'
heart['rest_ecg'][heart['rest_ecg'] == 1] = 'low'
heart['rest_ecg'][heart['rest_ecg'] == 2] = 'high'

heart['exercise_induced_angina'][heart['exercise_induced_angina'] == 0] = 'no'
heart['exercise_induced_angina'][heart['exercise_induced_angina'] == 1] = 'yes'

heart['st_slope'][heart['st_slope'] == 0] = 'upsloping'
heart['st_slope'][heart['st_slope'] == 1] = 'flat'
heart['st_slope'][heart['st_slope'] == 2] = 'downsloping'


heart['thalassemia'][heart['thalassemia'] == 0] = 'very low'
heart['thalassemia'][heart['thalassemia'] == 1] = 'low'
heart['thalassemia'][heart['thalassemia'] == 2] = 'high'
heart['thalassemia'][heart['thalassemia'] == 3] = 'very high'



heart.shape
heart.info()
null_columns=heart.columns[heart.isnull().any()]
heart[null_columns].isnull().sum()
heart.describe().T

def count_uni_plot(df):
    ax = sns.countplot(x=heart[df], data=heart)
    return ax

count_uni_plot('sex')
count_uni_plot('chest_pain_type')
count_uni_plot('fasting_blood_sugar')
count_uni_plot('rest_ecg')
count_uni_plot('exercise_induced_angina')
count_uni_plot('st_slope')
count_uni_plot('thalassemia')



def count_bi_plot(df):
    yes_disease = heart[heart['target']==1][df].value_counts()
    no_disease = heart[heart['target']==0][df].value_counts()
    out = pd.DataFrame([yes_disease, no_disease])
    out.index = ['yes_disease','no_disease']
    print(out)
    print('------------------------------------------------------------------------------------------------------------------------')
    out.plot(kind='bar',stacked=True, figsize=(10,6))
    return out 

count_bi_plot('sex')
count_bi_plot('chest_pain_type')
count_bi_plot('fasting_blood_sugar')
count_bi_plot('rest_ecg')
count_bi_plot('exercise_induced_angina')
count_bi_plot('st_slope')
count_bi_plot('thalassemia')
count_bi_plot('chol_bin')
count_bi_plot('trestbps_bin')
count_bi_plot('age_bin')


sex = pd.crosstab(heart['sex'],heart['target'])
cp = pd.crosstab(heart.chest_pain_type,heart.target)


##########################################
    #building model using logistic#
##########################################


import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

mydb = mysql.connector.connect(
host="localhost",
user="chandra",
passwd="chandra12v",
db='project')


sql_select_Query = "select * from heart"
cur = mydb.cursor()
cur.execute(sql_select_Query)
records = (cur.fetchall())

heart_model=pd.DataFrame(records)

heart_model.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']







X = heart_model.iloc[:,:-1]
Y = heart_model.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3)

scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scale)
X_test_scale =scale.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scale)




#applying logistic regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#applying naive baies
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#creating confusion matrix
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#finding the accuracy score
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

print('Accuracy of Naive baies classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

#precision alone
precision=precision_score(y_test, y_pred, average='macro') 
print(precision) 

#precision recall and f1 score
all=precision_recall_fscore_support(y_test, y_pred, average='macro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)


#accuracy
acc=accuracy_score(y_test, y_pred)
print(acc)

#MCC
mat=matthews_corrcoef(y_test, y_pred) 
print(mat)

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np

crossValScoreAccuracy=cross_val_score(classifier, X, Y,cv=10)
print("Mean Accuracy = ",np.mean(crossValScoreAccuracy))

precisionScorer = make_scorer(precision_score, pos_label='True')
crossValScorePrecision=cross_val_score(classifier,X,Y,cv=10,scoring=precisionScorer)
print("Mean Precision = ",np.mean(crossValScorePrecision))

recallScorer=make_scorer(classifier,pos_label='True')
crossValScoreRecall=cross_val_score(classifier,X,Y,cv=10,scoring=recallScorer)
print("Mean Recall = ",np.mean(crossValScoreRecall))




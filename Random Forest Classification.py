#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\marketing.csv')
data


# In[3]:


data.describe(include='all')


# In[4]:


data=data.drop('job', axis=1)
data=data.drop('month', axis=1)


# In[5]:


data.isnull().sum()


# In[6]:


data['result']=data['result'].map({'yes': 1, 'no': 0})


# In[7]:


data.head()


# In[8]:


data.corr()['result']


# In[9]:


data=data.drop('ID', axis=1)
data=data.drop('age', axis=1)
data=data.drop('balance', axis=1)
data=data.drop('day', axis=1)


# In[10]:


data.head()


# In[11]:


data.dtypes


# In[12]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[['campaign', 'pdays', 'previous']]
vif=pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# In[13]:


for i in data[['campaign', 'pdays', 'previous']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[14]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[15]:


for i in data[['campaign', 'pdays', 'previous']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i]) 


# In[16]:


for i in data[['campaign', 'pdays', 'previous']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[17]:


data=data.reset_index(drop=True)


# In[18]:


data.describe(include='all')


# In[19]:


data=pd.get_dummies(data, drop_first=True)


# In[20]:


data


# In[21]:


data.columns


# In[22]:


data=data[['campaign', 'pdays', 'previous', 'marital_married',
       'marital_single', 'education_secondary', 'education_tertiary',
       'education_unknown', 'default_yes', 'housing_yes', 'loan_yes',
       'contact_telephone', 'contact_unknown', 'response_other',
       'response_success', 'response_unknown', 'result']]


# In[23]:


data


# In[24]:


X=data.drop('result', axis=1)
y=data['result']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[27]:


y_pred=dtc.predict(X_test)


# In[28]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[29]:


accuracy_score(y_pred, y_test)


# In[30]:


confusion_matrix(y_pred, y_test)


# In[31]:


from sklearn.metrics import roc_auc_score


# In[32]:


roc_score=roc_auc_score(y_pred, y_test)
print('Roc_Auc_Score:', roc_score*100)


# In[33]:


print('Gini_Score:', (roc_score*2-1)*100)


# In[34]:


rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[35]:


y_predRFC=rfc.predict(X_test)


# In[36]:


confusion_matrix(y_predRFC, y_test)


# In[37]:


accuracy_score(y_predRFC, y_test)


# In[38]:


roc_score=roc_auc_score(y_predRFC, y_test)
print('Roc_Auc_Score:', roc_score*100)
print('Gini_Score:', (roc_score*2-1)*100)


# In[39]:


from sklearn.feature_selection import SelectFromModel


# In[40]:


sfm = SelectFromModel(rfc)
sfm.fit(X_train, y_train)


# In[41]:


selected_feature= X_train.columns[(sfm.get_support())]
selected_feature


# In[42]:


feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores


# In[43]:


X_train=X_train[['response_success', 'campaign', 'housing_yes', 'contact_unknown']]
X_test=X_test[['response_success', 'campaign', 'housing_yes', 'contact_unknown']]


# In[44]:


rfc_importance=RandomForestClassifier()
rfc_importance.fit(X_train, y_train)


# In[45]:


y_pred_importance=rfc_importance.predict(X_test)


# In[46]:


print('Model accuracy score with important features:', accuracy_score(y_test, y_pred_importance)*100)


# In[47]:


roc_score=roc_auc_score(y_pred_importance, y_test)
print('Roc_Auc_Score:', roc_score*100)
print('Gini_Score:', (roc_score*2-1)*100)


# In[48]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[49]:


rfc_randomized = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=1, random_state=42, n_jobs = -1)

rfc_randomized.fit(X_train, y_train)


# In[50]:


rfc_randomized.best_params_


# In[51]:


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    roc_score = roc_auc_score(y_test, y_prob)
    
    gini_score = roc_score*2-1
    
    print('Model Performance')

    print('Gini Score:', gini_score*100)
    
    return gini_score


# In[52]:


base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)


# In[53]:


optimized_model = rfc_randomized.best_estimator_
optmized_accuracy = evaluate(optimized_model, X_test, y_test)


# In[54]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

y_prob = base_model.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[55]:


y_prob = optimized_model.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[56]:


variables = []
train_gini_scores = []
test_gini_scores = []


for i in X_train.columns:
    X_train_single = X_train[[i]]
    X_test_single = X_test[[i]]

    
    optimized_model.fit(X_train_single, y_train)
    
    y_prob_train_single = optimized_model.predict_proba(X_train_single)[:, 1]

    train_roc_score = roc_auc_score(y_train, y_prob_train_single)
    train_gini_score = 2 * train_roc_score - 1

    
    y_prob_test_single = optimized_model.predict_proba(X_test_single)[:, 1]

    test_roc_score = roc_auc_score(y_test, y_prob_test_single)
    test_gini_score = 2 * test_roc_score - 1


    variables.append(i)
    train_gini_scores.append(train_gini_score)
    test_gini_scores.append(test_gini_score)


results_df = pd.DataFrame({
                            'Variable': variables,
                            'Train Gini': train_gini_scores,
                            'Test Gini': test_gini_scores
                        })

results_df_sorted = results_df.sort_values(by='Test Gini', ascending=False)

pd.options.display.float_format = '{:.4f}'.format

results_df_sorted


# In[57]:


X=data[['contact_unknown', 'response_success', 'housing_yes', 'campaign']]
y=data['result']


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[59]:


last_optimized= rfc_randomized.fit(X_train, y_train)

optmized_accuracy = evaluate(last_optimized, X_test, y_test)


# In[64]:


test_data=pd.read_excel(r'C:\Users\ASUS\Downloads\marketing_test.xlsx')


# In[65]:


test_data=test_data[['contact', 'response', 'housing', 'campaign']]


# In[66]:


test_data


# In[67]:


test_data=pd.get_dummies(test_data, drop_first=True)


# In[68]:


test_data


# In[69]:


X.columns


# In[70]:


test_data=test_data[['contact_unknown', 'response_success', 'housing_yes', 'campaign']]


# In[71]:


test_data


# In[72]:


X


# In[73]:


test_data['Prediction']=last_optimized.predict_proba(test_data)[:,1]


# In[74]:


test_data


# In[ ]:





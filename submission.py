__author__ = "whackadoodle"

import pandas as pd
import numpy as np
import pylab as P
import csv as csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#creating pandas dataframes
submissions_df = pd.read_csv('submissions.csv',header = 0)
problems_df = pd.read_csv('problems.csv',header = 0)
users_df = pd.read_csv('users.csv',header = 0)

#converting categorical variables from text to numeric values
submissions_df['solved'] = submissions_df['solved_status'].map({'AT':0,'SO':1,'UK':0}).astype(int)
submissions_df['results'] = submissions_df['result'].map({'AC':1,'PAC':0,'TLE':0,'CE':0,'RE':0,'WA':0}).astype(int)
problems_df['levels'] = problems_df['level'].map({'E':0,'E-M':1,'M':2,'M-H':3,'H':4})
problems_df = problems_df.drop(['level'],axis = 1)

#submissions_df.loc[submissions_df['solved'] == 1,'results'] = 1
#submissions_df.loc[submissions_df['solved'] == 0,'results'] = 0

#Data Munging on the 'levels' field of the problems dataframe using SVM
null_problems = problems_df[problems_df['levels'].isnull()]
null_problems_df = null_problems.drop(['problem_id','accuracy','rating','tag1','tag2','tag3','tag4','tag5','levels'],axis = 1)
non_null_problems = problems_df[problems_df['levels'].notnull()]
non_null_problems = non_null_problems.drop(['problem_id','accuracy','rating','tag1','tag2','tag3','tag4','tag5'],axis = 1)


train_data_problems = non_null_problems.values
test_data_problems = null_problems_df.values

clf = svm.SVC()
clf = clf.fit(train_data_problems[0::,:-1],train_data_problems[0::,2])
output = clf.predict(test_data_problems)
test_data_problems = np.insert(test_data_problems,2,output,axis = 1)
null_problems['levels'] = output

problems_df = problems_df.drop(['tag1','tag2','tag3','tag4','tag5'],axis = 1)
problems_df = problems_df.dropna()
null_problems = null_problems.drop(['tag1','tag2','tag3','tag4','tag5'],axis = 1)
problems_df = pd.concat([problems_df,null_problems])

#merging the submissions,users and problems dataframes
problem_submission_df = pd.merge(submissions_df,problems_df,on='problem_id')
submission_df = pd.merge(problem_submission_df,users_df,on='user_id')
submission_df = submission_df.drop(['language_used','skills','user_type','execution_time','result','solved_status'],axis = 1)

test_df = pd.read_csv('../test/test.csv')
users_test_df = pd.read_csv('../test/users.csv')
problems_test_df = pd.read_csv('../test/problems.csv')
#print test_df.info()
#print users_test_df.info()
#print problems_test_df.info()

#Dataframes for testing
users_test_df = users_test_df.drop(['skills','user_type'],axis = 1)
problems_test_df['levels'] = problems_test_df['level'].map({'E':0,'E-M':1,'M':2,'M-H':3,'H':4})
problems_test_df = problems_test_df.drop(['level'],axis = 1)
problems_test_df = problems_test_df.drop(['tag1','tag2','tag3','tag4','tag5'],axis = 1)

#Data munging on the test data
null_problems_test = problems_test_df[problems_test_df['levels'].isnull()]
null_problems_df_test = null_problems_test.drop(['problem_id','accuracy','rating','levels'],axis = 1)

test_data_problems_test = null_problems_df_test.values
output_test = clf.predict(test_data_problems_test)
test_data_problems_test = np.insert(test_data_problems_test,2,output_test,axis = 1)
null_problems_test['levels'] = output_test
problems_test_df = problems_test_df.dropna()
problems_test_df = pd.concat([problems_test_df,null_problems_test])

#merging the submissions,users and problems test dataframes
test_df = pd.merge(test_df,problems_test_df,on = 'problem_id')
test_df = pd.merge(test_df,users_test_df,on = 'user_id')

submission_df = submission_df[['user_id','problem_id','accuracy','solved_count_x','error_count','rating','levels','solved_count_y','attempts','results','solved']]

id_array = test_df['Id'].values
test_df = test_df.drop(['Id'],axis = 1)
train_data = submission_df.values
test_data = test_df.values

#training and testing w.r.t 'results'
#forest = RandomForestClassifier(n_estimators=100)
#forest = forest.fit(train_data[0::,:-2], train_data[0::,9])
#result_output = forest.predict(test_data).astype(int)


#training and testing w.r.t 'solved' attribute 
#Random Forest used for classification
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::,:-2], train_data[0::,10])
solved_output = forest.predict(test_data).astype(int)

'''solved_status = []
i = 0
while i<len(result_output):
	if(result_output[i] == 1 & solved_output[i] == 1):
		solved_status.append(1)
	else:
		solved_status.append(0)
	i = i+1
solved_status = np.array(solved_status)
'''

#writing the predicted output to the csv file
predictions_file = open("results.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","solved_status"])
open_file_object.writerows(zip(id_array,solved_output))
predictions_file.close()


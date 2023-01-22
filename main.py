import pandas as pd
from utils import DataTools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

utils = DataTools()
pd.set_option('display.max_columns', None)
sourcedf = pd.read_csv('subjects-questions.csv')
print('Full dataset overview:\n {}\n'.format(sourcedf.head()))
subjects_list = [''] * 3
cases_list = [''] * 3
data = ['Full', 'Train', 'Test']
subjects_list[0], cases_list[0] = utils.overview_data(sourcedf, data[0])
utils.plot_word_distribution(sourcedf, subjects_list[0], max_words=25)
# X_train, X_test, y_train, y_test = train_test_split(sourcedf.iloc[:, :-1], sourcedf['Subject'], test_size=0.2,
#                                                     shuffle=True, stratify=sourcedf['Subject'], random_state=0)
# subjects_list[1], cases_list[1] = utils.overview_data(y_train, data[1])
# subjects_list[2], cases_list[2] = utils.overview_data(y_test, data[2])
# utils.pie_plot(subjects_list, cases_list, data)

# Grid search and model optimization
# algorithm = ['logistic regression']
# preprocess = ['count']
# scoring = 'accuracy'
# params = [
#     {'classifier': [], 'preprocess': [],
#      'preprocess__min_df': [1], 'preprocess__max_df': [1.0], 'preprocess__ngram_range': [(1, 1)],
#      'classifier__C': [1]}]
# grid = utils.cross_grid_validation(algorithm, preprocess, params,
#                                    X_train['eng'], y_train, X_test['eng'], y_test, scoring, 5)
# pd_grid = pd.DataFrame(grid.cv_results_)
# print(pd_grid)




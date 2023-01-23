import pandas as pd
import utils
from data_visualization import DataPlots
from sklearn.model_selection import train_test_split


max_df = 1.0
min_df = 1
stop_words = None

visualization = DataPlots()
pd.set_option('display.max_columns', None)
sourcedf = pd.read_csv('subjects-questions.csv')
print('Full dataset overview:\n {}\n'.format(sourcedf.head()))
subjects_list = [''] * 3
cases_list = [''] * 3
data = ['Full', 'Train', 'Test']
subjects_list[0], cases_list[0] = utils.overview_data(sourcedf, data[0])
#visualization.plot_word_distribution(sourcedf, subjects_list[0], min_df=min_df, max_df=max_df, stop_words=stop_words,
#                                     tokenizer=None)
# visualization.plot_length_distribution(sourcedf, subjects_list[0], min_df=min_df, max_df=max_df, stop_words=stop_words)
#visualization.show_tfid_distribution(sourcedf, min_df=min_df, max_df=max_df, stop_words=stop_words)

X_train, X_test, y_train, y_test = train_test_split(sourcedf['eng'], sourcedf['Subject'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Subject'], random_state=0)

subjects_list[1], cases_list[1] = utils.overview_data(y_train, data[1])
subjects_list[2], cases_list[2] = utils.overview_data(y_test, data[2])
# visualization.pie_plot(subjects_list, cases_list, data)

# Coefficients analysis
algorithm = ['logistic regression']
preprocess = ['count']
scoring = 'accuracy'
params = [
    {'classifier': [], 'preprocess': [],
     'preprocess__stop_words': ['english'],
     'classifier__C': [1]}]
grid = utils.cross_grid_validation(algorithm, preprocess, params, X_train, y_train, X_test, y_test,
                                   scoring, visualization, subjects_list[1], 5)

# Grid search and model optimization
# algorithm = ['logistic regression']
# preprocess = ['count']
# scoring = 'accuracy'
# params = [
#     {'classifier': [], 'preprocess': [],
#      'preprocess__stop_words': ['english'], 'preprocess__min_df': [1], 'preprocess__max_df': [1.0],
#      'preprocess__ngram_range': [(1, 1)], 'classifier__C': [1]}]
# grid = utils.cross_grid_validation(algorithm, preprocess, params,
#                                    X_train['eng'], y_train, X_test['eng'], y_test, scoring, 5)
# pd_grid = pd.DataFrame(grid.cv_results_)
# print(pd_grid)

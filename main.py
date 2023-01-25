import pandas as pd
import utils
from data_visualization import DataPlots
from sklearn.model_selection import train_test_split


visualization = DataPlots()
pd.set_option('display.max_columns', None)
sourcedf = pd.read_csv('subjects-questions.csv')
print('Full dataset overview:\n {}\n'.format(sourcedf.head()))
subjects_list = [''] * 3
cases_list = [''] * 3
data = ['Full', 'Train', 'Test']
subjects_list[0], cases_list[0] = utils.overview_data(sourcedf, data[0])
#visualization.plot_word_distribution(sourcedf, subjects_list[0], min_df=1, max_df=1.0, stop_words='english',
#                                     tokenizer=None)
# visualization.plot_length_distribution(sourcedf, subjects_list[0], min_df=1, max_df=1.0, stop_words='english')
#visualization.show_tfid_distribution(sourcedf, min_df=1, max_df=1.0, stop_words=None)

X_train, X_test, y_train, y_test = train_test_split(sourcedf['eng'], sourcedf['Subject'], test_size=0.2,
                                                    shuffle=True, stratify=sourcedf['Subject'], random_state=0)

subjects_list[1], cases_list[1] = utils.overview_data(y_train, data[1])
subjects_list[2], cases_list[2] = utils.overview_data(y_test, data[2])
# visualization.pie_plot(subjects_list, cases_list, data)

# Coefficients analysis
#utils.logreg_coeffs_comparison(X_train, y_train, visualization, subjects_list[1], C=1, ngrams=1)

# Grid search and model optimization
scoring = 'accuracy'
params = [
    {'classifier': ['logistic regression'], 'preprocess': ['count', 'tfidf'],
     'preprocess__stop_words': [None, 'english'], 'preprocess__min_df': [1, 2, 3, 4], 'preprocess__max_df': [1.0],
     'preprocess__ngram_range': [(1, 2)], 'classifier__C': [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5]},
    {'classifier': ['linear svc'], 'preprocess': ['count', 'tfidf'],
     'preprocess__stop_words': [None, 'english'], 'preprocess__min_df': [1, 2, 3, 4], 'preprocess__max_df': [1.0],
     'preprocess__ngram_range': [(1, 2)], 'classifier__C': [0.05, 0.075, 0.1, 0.25, 0.5, 0.75]},
    {'classifier': ['multinomial'], 'preprocess': ['count', 'tfidf'],
     'preprocess__stop_words': [None, 'english'], 'preprocess__min_df': [1, 2, 3, 4], 'preprocess__max_df': [1.0],
     'preprocess__ngram_range': [(1, 2)], 'classifier__alpha': [0.00005, 0.0001, 0.00035, 0.0007, 0.001, 0.0015]}]
grid = utils.cross_grid_validation(params, X_train, y_train, X_test, y_test, scoring, 5)
pd_grid = pd.DataFrame(grid.cv_results_)
print(pd_grid)
utils.param_sweep_matrix(visualization, params=pd_grid['params'], test_score=pd_grid['mean_test_score'])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import numpy as np


def overview_data(df, name):
    cases = []
    try:
        print('The {} dataset have {} samples with {} columns'.format(name.lower(), df.shape[0], df.shape[1]))
        subjects = np.sort(df['Subject'].unique())
    except IndexError:
        print('The {} dataset have {} samples and single column'.format(name.lower(), df.shape[0]))
        subjects = np.sort(df.unique())
    print('The number of subjects in the full dataset are {} corresponding to {}'.format(len(subjects), subjects))
    for subject in subjects:
        try:
            cases.append(len(df.loc[df['Subject'] == subject]))
            print('Subject ' + subject.upper() + ' has ' + str(len(df.loc[df['Subject'] == subject])) + ' samples')
        except KeyError:
            cases.append(len(df[df == subject]))
            print('Subject ' + subject.upper() + ' has ' + str(len(df[df == subject])) + ' samples')
    print('\n')
    return subjects, cases


def cross_grid_validation(param_grid, X_train, y_train, X_test, y_test, scoring, nfolds=5):
    time0 = time.time()
    model = []
    preprocess = []
    for i in range(len(param_grid)):
        model = []
        for j in range(len(param_grid[i]['classifier'])):
            model.append(create_model(param_grid[i]['classifier'][j]))
        param_grid[i]['classifier'] = model
        preproc = []
        for j in range(len(param_grid[i]['preprocess'])):
            preproc.append(create_preprocess(param_grid[i]['preprocess'][j]))
        param_grid[i]['preprocess'] = preproc
    pipe = Pipeline([('preprocess', preprocess), ('classifier', model)])
    grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring=scoring)
    grid_search.fit(X_train, y_train)
    print('Best estimator preprocessing: {}'.format(str(grid_search.best_estimator_.named_steps['preprocess'])))
    print('Best estimator classifier: {}\n'.format(str(grid_search.best_estimator_.named_steps['classifier'])))
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))
    print('Test set score: {:.4f}'.format(grid_search.score(X_test, y_test)))
    print('Grid search time: {:.1f}\n'.format(time.time() - time0))
    return grid_search


def logreg_coeffs_comparison(X_train, y_train, dataplot, subjects, ngrams=1, C=1):
    param_grid = [{'classifier': [], 'preprocess': [],
                  'preprocess__stop_words': ['english'], 'preprocess__ngram_range': [(1, ngrams)], 'classifier__C': []}]
    model = create_model('logistic regression')
    param_grid[0]['classifier'] = [model]
    param_grid[0]['classifier__C'] = [C]
    for i in range(2):
        if i == 0:
            preprocess = create_preprocess('count')
        else:
            preprocess = create_preprocess('tfidf')
        param_grid[0]['preprocess'] = [preprocess]
        pipe = Pipeline([('preprocess', preprocess), ('classifier', model)])
        grid_search = GridSearchCV(pipe, param_grid, cv=2)
        grid_search.fit(X_train, y_train)
        feature_names = grid_search.best_estimator_.named_steps['preprocess'].get_feature_names_out()
        coeffs = grid_search.best_estimator_.named_steps['classifier'].coef_
        for j in range(coeffs.shape[0]):
            max_coeffs = np.argsort(-coeffs[j, :])
            min_coeffs = np.argsort(coeffs[j, :])
            dataplot.plot_logreg_coeffs(coeffs[j], max_coeffs, min_coeffs, feature_names,
                                        subjects[j], C, preprocess, ngrams)


def create_preprocess(pre):
    if 'norm' in pre.lower():
        preprocess = MinMaxScaler()
    elif 'std' in pre.lower() or 'standard' in pre.lower():
        preprocess = StandardScaler()
    elif 'count' in pre.lower():
        preprocess = CountVectorizer()
    elif 'tfidf' in pre.lower():
        preprocess = TfidfVectorizer(norm=None)
    else:
        preprocess = None
        print('WARNING: no preprocessor was selected\n')
    return preprocess


def create_model(algorithm):
    if algorithm.lower() == 'knn':
        model = KNeighborsClassifier()
    elif 'logistic' in algorithm.lower() or 'regression' in algorithm.lower() or 'logreg' in algorithm.lower():
        model = LogisticRegression(random_state=0)
    elif 'linear' in algorithm.lower() or 'svc' in algorithm.lower():
        model = LinearSVC(random_state=0, dual=False)
    elif 'gaussian' in algorithm.lower():
        model = GaussianNB()
    elif 'multinomial' in algorithm.lower():
        model = MultinomialNB()
    elif algorithm.lower() == 'tree':
        model = DecisionTreeClassifier(random_state=0)
    elif algorithm.lower() == 'forest' or algorithm.lower() == 'random':
        model = RandomForestClassifier(random_state=0)
    elif 'gradient' in algorithm.lower() or 'boosting' in algorithm.lower():
        model = GradientBoostingClassifier(random_state=0)
    elif algorithm.lower() == 'svm':
        model = SVC(random_state=0)
    elif algorithm.lower() == 'mlp':
        model = MLPClassifier(random_state=0)
    else:
        print('\nERROR: Algorithm was NOT provided. Note the type must be a list.\n')
        model = None
    return model


def param_sweep_matrix(dataplot, params, test_score):
    """Postprocess the cross validation grid search results and plot the parameter sweep"""
    models = []
    model = ''
    for i in range(len(params)):
        for j in range(len(params[i])):
            string = str(params[j]['classifier'])
            for k in range(len(str(string))):
                if string[k] == '(':
                    model = string[:k]
                    break
            else:
                continue
            break
        if model not in models:
            models.append(model)
    for m in range(len(models)):
        test = []
        feat_name = []
        feat = []
        for i in range(len(params)):
            string = str(params[i]['classifier'])
            for k in range(len(str(string))):
                if string[k] == '(':
                    model = string[:k]
                    break
            if model == models[m]:
                test.append(test_score[i])
                for key, value in params[i].items():
                    if key == 'preprocess':
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                val = str(value)[:str(value).index('(')]
                                feat[k].append(val)
                                break
                    if 'classifier__' in key:
                        key = key.replace('classifier__', '')
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                feat[k].append(value)
                                break
                    if 'preprocess__' in key:
                        key = key.replace('preprocess__', '')
                        if key not in feat_name:
                            feat_name.append(key)
                            feat.append([])
                        for k in range(len(feat_name)):
                            if feat_name[k] == key:
                                feat[k].append(value)
                                break
        feat_name_unique = {}
        feat_name_sweep = []
        feat_sweep = []
        feat_index = []
        for i in range(len(feat)):
            unique = []
            [unique.append(feat[i][x]) for x in range(len(feat[i])) if feat[i][x] not in unique]
            if len(unique) == 1:
                feat_name_unique[feat_name[i]] = unique[0]
            elif len(unique) > 1:
                feat_name_sweep.append(feat_name[i])
                feat_sweep.append(unique)
                feat_index.append(i)
        if len(feat_sweep) == 0:
            test_matrix = np.array([test])
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique)
        if len(feat_sweep) == 1:
            test_matrix = np.array([test])
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                       xtick=feat_sweep[0], xtag=feat_name_sweep[0])
        if len(feat_sweep) == 2:
            test_matrix = np.zeros([len(feat_sweep[0]), len(feat_sweep[1])])
            for j in range(len(feat_sweep[0])):
                for h in range(len(feat_sweep[1])):
                    test_index = 0
                    for r in range(len(test)):
                        if feat_sweep[0][j] == feat[feat_index[0]][r] and feat_sweep[1][h] == feat[feat_index[1]][r]:
                            test_index = r
                            break
                    test_matrix[j, h] = test[test_index]
            dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                       xtick=feat_sweep[1], xtag=feat_name_sweep[1],
                                       ytick=feat_sweep[0], ytag=feat_name_sweep[0])
        if len(feat_sweep) == 3:
            feat_sweep_bu = feat_sweep.copy()
            feat_name_sweep_bu = feat_name_sweep.copy()
            feat_index_bu = feat_index.copy()
            dims = []
            for i in range(len(feat_sweep)):
                dims.append(len(feat_sweep[i]))
            if min(dims) == dims[1]:
                feat_sweep[0] = feat_sweep_bu[1]
                feat_sweep[1] = feat_sweep_bu[0]
                feat_name_sweep[0] = feat_name_sweep_bu[1]
                feat_name_sweep[1] = feat_name_sweep_bu[0]
                feat_index[0] = feat_index_bu[1]
                feat_index[1] = feat_index_bu[0]
            elif min(dims) == dims[2]:
                feat_sweep[0] = feat_sweep_bu[2]
                feat_sweep[2] = feat_sweep_bu[0]
                feat_name_sweep[0] = feat_name_sweep_bu[2]
                feat_name_sweep[2] = feat_name_sweep_bu[0]
                feat_index[0] = feat_index_bu[2]
                feat_index[2] = feat_index_bu[0]
            test_matrix = np.zeros([len(feat_sweep[1]), len(feat_sweep[2]), len(feat_sweep[0])])
            for p in range(len(feat_sweep[0])):
                for j in range(len(feat_sweep[1])):
                    for h in range(len(feat_sweep[2])):
                        test_index = 0
                        for r in range(len(test)):
                            if feat_sweep[2][h] == feat[feat_index[2]][r] \
                                    and feat_sweep[1][j] == feat[feat_index[1]][r]\
                                    and feat_sweep[0][p] == feat[feat_index[0]][r]:
                                test_index = r
                                break
                        test_matrix[j, h, p] = test[test_index]
                dataplot.plot_params_sweep(models[m], test_matrix, feat_name_unique,
                                           xtick=feat_sweep[2], xtag=feat_name_sweep[2],
                                           ytick=feat_sweep[1], ytag=feat_name_sweep[1],
                                           ztick=feat_sweep[0], ztag=feat_name_sweep[0])

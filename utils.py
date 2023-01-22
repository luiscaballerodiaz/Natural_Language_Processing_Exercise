import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
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
import math
import numpy as np


class DataTools:
    """Class to plot and apply tools to dataset"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.plots_row = 2

    def overview_data(self, df, name):
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

    def pie_plot(self, subjects, cases, plots):
        fig, axes = plt.subplots(1, len(cases), figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i in range(len(cases)):
            explode = [0.1] * len(cases[i])
            ax[i].pie(x=cases[i], explode=explode, labels=subjects[i], autopct=self.make_autopct(cases[i]),
                      shadow=True, textprops={'fontsize': 16})
            ax[i].set_title(plots[i] + ' data pie plot', fontsize=20, fontweight='bold')
            ax[i].legend()
        fig.tight_layout()
        fig.suptitle('Pie plot for subject distribution', fontsize=24, fontweight='bold')
        plt.savefig('Subjects pie plot.png', bbox_inches='tight')
        plt.clf()

    @staticmethod
    def make_autopct(values):
        def my_autopct(pct):
            val = int(round(pct * sum(values) / 100.0))
            return '{}\n{:.1f}%'.format(val, pct)
        return my_autopct

    def cross_grid_validation(self, algorithm, pre, param_grid, X_train, y_train, X_test, y_test, scoring, nfolds=5):
        time0 = time.time()
        model = []
        preprocess = []
        for i in range(len(algorithm)):
            model.append(self.create_model(algorithm[i]))
            preprocess.append(self.create_preprocess(pre[i]))
            param_grid[i]['classifier'] = [model[i]]
            param_grid[i]['preprocess'] = [preprocess[i]]
        pipe = Pipeline([('preprocess', preprocess), ('classifier', model)])
        grid_search = GridSearchCV(pipe, param_grid, cv=nfolds, scoring=scoring)
        grid_search.fit(X_train, y_train)
        vect = grid_search.best_estimator_.named_steps['preprocess']
        if 'vectorizer' in str(vect).lower():
            vect.fit(X_train)
            feature_names = vect.get_feature_names_out()
            print('Vectorizer to assess the vocabulary:\n{}'.format(str(vect)))
            print('Number of features in the bag of words: {}'.format(len(feature_names)))
            print('First 50 features:\n{}\n'.format(feature_names[:50]))
        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best cross-validation score: {:.4f}'.format(grid_search.best_score_))
        print('Test set score: {:.4f}'.format(grid_search.score(X_test, y_test)))
        print('Grid search time: {:.1f}\n'.format(time.time() - time0))
        return grid_search

    @staticmethod
    def create_preprocess(pre):
        if 'norm' in pre.lower():
            preprocess = MinMaxScaler()
        elif 'std' in pre.lower() or 'standard' in pre.lower():
            preprocess = StandardScaler()
        elif 'count' in pre.lower():
            preprocess = CountVectorizer()
        elif 'tfid' in pre.lower():
            preprocess = TfidfVectorizer(norm=None)
        else:
            preprocess = None
        return preprocess

    @staticmethod
    def create_model(algorithm):
        if algorithm.lower() == 'knn':
            model = KNeighborsClassifier()
        elif 'logistic' in algorithm.lower() or 'regression' in algorithm.lower() or 'logreg' in algorithm.lower():
            model = LogisticRegression(random_state=0)
        elif 'linear' in algorithm.lower() or 'svc' in algorithm.lower():
            model = LinearSVC(random_state=0, dual=False)
        elif 'naive' in algorithm.lower() or 'bayes' in algorithm.lower():
            model = GaussianNB()
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

    def plot_word_distribution(self, data, subjects, max_words=50):
        """Plot the word distribution in the dataset"""
        # Total word counting
        vect = CountVectorizer()
        vect.fit(data['eng'])
        bag_of_words = vect.transform(data['eng'])
        words = bag_of_words.toarray().shape[1]
        print('Number of words in the full dataset: {}'.format(words))
        repeat = np.sum(bag_of_words.toarray(), axis=0)
        features = vect.get_feature_names_out()
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        xtick = [''] * max_words
        for i in range(max_words):
            index_max = np.argmax(repeat)
            ax.bar(i + 1, repeat[index_max], color='b', width=self.bar_width, edgecolor='black')
            xtick[i] = features[index_max]
            repeat[index_max] = 0
        ax.set_xticks(range(1, max_words + 1), xtick, ha='center', rotation=50)
        ax.set_title('Most common words in the full dataset', fontsize=24, fontweight='bold')
        ax.set_xlabel('Words', fontweight='bold', fontsize=14)
        ax.set_ylabel('Occurrences', fontweight='bold', fontsize=14)
        ax.grid()
        fig.tight_layout()
        plt.savefig('Word counting dataset.png', bbox_inches='tight')
        plt.clf()

        # Word counting per subject
        repeat_subject = [''] * len(subjects)
        features_subject = [''] * len(subjects)
        xtick_subject = [''] * len(subjects)
        words = []
        for i, subject in zip(range(len(subjects)), subjects):
            vect = CountVectorizer()
            data_filter = data['eng'].loc[(data['Subject'] == subject)]
            vect.fit(data_filter)
            bag_of_words = vect.transform(data_filter)
            words.append(bag_of_words.toarray().shape[1])
            print('Number of words in questions for subject {}: {}'.format(subject, words[i]))
            repeat_subject[i] = np.sum(bag_of_words.toarray(), axis=0)
            features_subject[i] = vect.get_feature_names_out()
            xtick_subject[i] = [''] * max_words
        fig, axes = plt.subplots(math.ceil(len(subjects) / self.plots_row), self.plots_row,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = self.plots_row - len(subjects) % self.plots_row
        if spare_axes == self.plots_row:
            spare_axes = 0
        for axis in range(self.plots_row - 1, self.plots_row - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(len(subjects) / self.plots_row) - 1, axis])
        ax = axes.ravel()
        for i in range(max_words):
            for j in range(len(subjects)):
                index_max = np.argmax(repeat_subject[j])
                ax[j].bar(i + 1, repeat_subject[j][int(index_max)], color='b', width=self.bar_width, edgecolor='black')
                xtick_subject[j][i] = features_subject[j][int(index_max)]
                repeat_subject[j][index_max] = 0
        for j in range(len(subjects)):
            ax[j].set_xticks(range(1, max_words + 1), xtick_subject[j], ha='center', rotation=50)
            ax[j].set_title(subjects[j].upper() + ' - total words ' + str(words[j]),
                            fontsize=18, fontweight='bold')
            ax[j].set_xlabel('Words', fontweight='bold', fontsize=14)
            ax[j].set_ylabel('Occurrences', fontweight='bold', fontsize=14)
            ax[j].grid()
        fig.suptitle('Most common words per each subject', fontsize=24, fontweight='bold')
        fig.tight_layout()
        plt.savefig('Word counting per subject.png', bbox_inches='tight')
        plt.clf()

        # Word sharing subject matrix


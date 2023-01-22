import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np


class DataPlots:
    """Class to plot and apply tools to dataset"""

    def __init__(self):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.subplots_row = 2

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

    def plot_word_distribution(self, data, subjects, min_df=1, max_df=1.0, stop_words='english', max_words=50):
        """Plot the word distribution in the dataset"""
        # Total word counting
        vect = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
        vect.fit(data['eng'])
        bag_of_words = vect.transform(data['eng'])
        words = bag_of_words.toarray().shape[1]
        print('Number of words in the full dataset: {}'.format(words))
        print('Shape of bag of words: {}'.format(bag_of_words.toarray().shape))
        repeat = np.sum(bag_of_words.toarray(), axis=0)
        features = vect.get_feature_names_out()
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        xtick = [''] * max_words
        for i in range(max_words):
            index_max = np.argmax(repeat)
            ax.bar(i + 1, repeat[index_max], color='b', width=self.bar_width, edgecolor='black')
            xtick[i] = features[index_max]
            repeat[index_max] = 0
        ax.set_xticks(range(1, max_words + 1), xtick, ha='center', rotation=90)
        ax.set_title('Most common words in the full dataset max_df = ' + str(max_df) + ', mind_df = ' + str(min_df) +
                     ' and stop_words = ' + str(stop_words), fontsize=24, fontweight='bold')
        ax.set_xlabel('Words (Total words = ' + str(words) + ')', fontweight='bold', fontsize=14)
        ax.set_ylabel('Occurrences', fontweight='bold', fontsize=14)
        ax.grid()
        fig.tight_layout()
        plt.savefig('Word counting dataset max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                    ' stop_words=' + str(stop_words) + '.png', bbox_inches='tight')
        plt.clf()

        # Word counting per subject
        max_words = round(max_words / 2)
        repeat_subject = [''] * len(subjects)
        features_subject = [''] * len(subjects)
        xtick_subject = [''] * len(subjects)
        words = []
        for i, subject in zip(range(len(subjects)), subjects):
            data_filter = data['eng'].loc[(data['Subject'] == subject)]
            vect.fit(data_filter)
            bag_of_words = vect.transform(data_filter)
            words.append(bag_of_words.toarray().shape[1])
            print('Number of words in questions for subject {}: {}'.format(subject, words[i]))
            repeat_subject[i] = np.sum(bag_of_words.toarray(), axis=0)
            features_subject[i] = vect.get_feature_names_out()
            xtick_subject[i] = [''] * max_words
        fig, axes = plt.subplots(math.ceil(len(subjects) / self.subplots_row), self.subplots_row,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = self.subplots_row - len(subjects) % self.subplots_row
        if spare_axes == self.subplots_row:
            spare_axes = 0
        for axis in range(self.subplots_row - 1, self.subplots_row - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(len(subjects) / self.subplots_row) - 1, axis])
        ax = axes.ravel()
        for i in range(max_words):
            for j in range(len(subjects)):
                index_max = np.argmax(repeat_subject[j])
                ax[j].bar(i + 1, repeat_subject[j][int(index_max)], color='b', width=self.bar_width, edgecolor='black')
                xtick_subject[j][i] = features_subject[j][int(index_max)]
                repeat_subject[j][index_max] = 0
        for j in range(len(subjects)):
            ax[j].set_xticks(range(1, max_words + 1), xtick_subject[j], ha='center', rotation=90)
            ax[j].set_title(subjects[j].upper() + ' - total words ' + str(words[j]),
                            fontsize=18, fontweight='bold')
            ax[j].set_xlabel('Words', fontweight='bold', fontsize=14)
            ax[j].set_ylabel('Occurrences', fontweight='bold', fontsize=14)
            ax[j].grid()
        fig.suptitle('Most common words per each subject max_df = ' + str(max_df) + ', mind_df = ' + str(min_df) +
                     ' and stop_words = ' + str(stop_words), fontsize=24, fontweight='bold')
        fig.tight_layout()
        plt.savefig('Word counting subject max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                    ' stop_words=' + str(stop_words) + '.png', bbox_inches='tight')
        plt.clf()

        # Word sharing subject matrix
        test_matrix = np.zeros([len(subjects), len(subjects)])
        for i in range(len(subjects)):
            for j in range(len(subjects)):
                test_matrix[i, j] = len(set(features_subject[i]) & set(features_subject[j]))
        features_all_subjects = len(set(features_subject[0]) & set(features_subject[1]) & set(features_subject[2])
                                    & set(features_subject[3]))
        print('\nFeatures in all subjects: {}'.format(features_all_subjects))
        exclusive_features = [len(set(features_subject[0]) - set(features_subject[1]) - set(features_subject[2])
                                  - set(features_subject[3])),
                              len(set(features_subject[1]) - set(features_subject[0]) - set(features_subject[2])
                                  - set(features_subject[3])),
                              len(set(features_subject[2]) - set(features_subject[1]) - set(features_subject[0])
                                  - set(features_subject[3])),
                              len(set(features_subject[3]) - set(features_subject[1]) - set(features_subject[2])
                                  - set(features_subject[0]))]
        for i in range(len(subjects)):
            print('Features exclusive in {}: {} ({:.1f}% of total subject words)'
                  .format(subjects[i], exclusive_features[i], 100 * exclusive_features[i] / len(features_subject[i])))
        print('\n')
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(test_matrix, cmap=plt.cm.PuBuGn)
        plt.colorbar()
        ax.set_title('Common words shared between subjects max_df = ' + str(max_df) + ', mind_df = ' + str(min_df) +
                     ' and stop_words = ' + str(stop_words), fontsize=24, fontweight='bold')
        ax.set_xticks(np.arange(0.5, len(subjects) + 0.5), subjects, fontsize=14, fontweight='bold')
        ax.set_yticks(np.arange(0.5, len(subjects) + 0.5), subjects, fontsize=14, fontweight='bold')
        for i in range(len(subjects)):
            for j in range(len(subjects)):
                ax.text(i + 0.5, j + 0.5, str(test_matrix[i, j]),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=10)
        fig.tight_layout(h_pad=2)
        plt.savefig('Word sharing subject max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                    ' stop_words=' + str(stop_words) + '.png', bbox_inches='tight')
        plt.clf()

    def plot_length_distribution(self, data, subjects, min_df=1, max_df=1.0, stop_words='english'):
        """Plot the word distribution in the dataset"""
        # Total length counting
        vect = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
        vect.fit(data['eng'])
        bag_of_words = vect.transform(data['eng'])
        lengths = np.sum(bag_of_words.toarray(), axis=1)
        bins = np.arange(0, max(lengths) + 11, 10)
        print('Shape of sample word lengths: {}'.format(lengths.shape))
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        counts, edges, bars = plt.hist(lengths, histtype='bar', bins=bins, alpha=0.25, color='b')
        plt.bar_label(bars)
        ax.set_title('Length sample distribution in words max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                     ' stop_words=' + str(stop_words), fontsize=24, fontweight='bold')
        ax.grid(visible=True)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_xlabel('Sample word length', fontsize=14)
        fig.tight_layout()
        plt.savefig('Length distribution dataset max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                    ' stop_words=' + str(stop_words) + '.png', bbox_inches='tight')
        plt.clf()

        # # Word counting per subject
        length_subject = [''] * len(subjects)
        for i, subject in zip(range(len(subjects)), subjects):
            data_filter = data['eng'].loc[(data['Subject'] == subject)]
            vect.fit(data_filter)
            bag_of_words = vect.transform(data_filter)
            length_subject[i] = np.sum(bag_of_words.toarray(), axis=1)
            print('Number of sample word lengths for subject {}: {}'.format(subject, len(length_subject[i])))
        fig, axes = plt.subplots(math.ceil(len(subjects) / self.subplots_row), self.subplots_row,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = self.subplots_row - len(subjects) % self.subplots_row
        if spare_axes == self.subplots_row:
            spare_axes = 0
        for axis in range(self.subplots_row - 1, self.subplots_row - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(len(subjects) / self.subplots_row) - 1, axis])
        ax = axes.ravel()
        for i in range(len(subjects)):
            counts, edges, bars = ax[i].hist(length_subject[i], histtype='bar', bins=bins, alpha=0.25, color='b')
            ax[i].bar_label(bars)
            ax[i].set_title(subjects[i].upper() + ' - length word distribution', fontsize=18, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_ylabel('Frequency', fontsize=14)
            ax[i].set_xlabel('Sample word length', fontsize=14)
        fig.suptitle('Length sample distribution per subject max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                     ' stop_words=' + str(stop_words), fontsize=24)
        fig.tight_layout()
        plt.savefig('Length distribution subject max_df=' + str(max_df) + ' mind_df=' + str(min_df) +
                    ' stop_words=' + str(stop_words) + '.png', bbox_inches='tight')
        plt.clf()

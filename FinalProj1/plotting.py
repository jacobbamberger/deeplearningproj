import matplotlib.pyplot as plt
import numpy as np


# Modified from the example on  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

def plot_error_bars(train_errors, test_errors, model_labels, save_path=None, train_stds=None, test_stds=None, title= None):

    if title is None:
        title = 'Comparison of different architectures'
    x = np.arange(len(model_labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()#
    rects1 = ax.bar(x - width/2, train_errors, width,  yerr=train_stds, label='Train error')
    rects2 = ax.bar(x + width/2, test_errors, width, yerr=test_stds, label='Test error')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Fraction of misclassified samples')


    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()

    # Write percentages on top
    #TODO: Uncomment for submission, this requires a very new version of matplotlib
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
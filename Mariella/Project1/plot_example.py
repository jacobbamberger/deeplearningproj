
import plot_snippet
import numpy as np


train_errors = [0.1, 0.2]
test_errors = [0.15, 0.22]

model_labels = ['A1', 'A2']

plot_snippet.plot_error_bars(train_errors, test_errors, model_labels)

plot_snippet.plot_error_bars(train_errors, test_errors, model_labels, 'figure.png')
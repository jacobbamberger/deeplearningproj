
import plot_snippet

train_errors = [0.1, 0.2]
train_stds = [0.01, 0.01]

test_errors = [0.15, 0.22]
test_stds = [0.02, 0.03]

model_labels = ['A1', 'A2']

plot_snippet.plot_error_bars(train_errors, test_errors, model_labels)

plot_snippet.plot_error_bars(train_errors, test_errors, model_labels, 'figure.png', train_stds, test_stds)
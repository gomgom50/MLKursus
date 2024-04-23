from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
import keras.backend as K
import matplotlib.pyplot as plt


def split_data(data):
    # Split data into features and target
    Y = data['EstimatedSalary']
    X = data.drop('EstimatedSalary', axis=1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class GradientMonitor(Callback):
    def on_train_begin(self, logs=None):
        self.gradients = []

    def on_batch_end(self, batch, logs=None):
        # Get the current training data
        training_data = self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights

        # Assume you have a single loss computed in the model
        loss = self.model.total_loss

        # Compute the gradients of the trainable weights wrt the loss
        gradients = self.model.optimizer.get_gradients(loss, self.model.trainable_weights)

        # The function to compute the gradients
        get_gradients = K.function(inputs=training_data, outputs=gradients)

        # Execute the function to compute the gradients
        inputs = [self.model._feed_inputs, self.model._feed_targets, self.model._feed_sample_weights]
        computed_gradients = get_gradients(inputs)

        # Store or process the gradients as needed
        self.gradients.append(computed_gradients)

    def on_epoch_end(self, epoch, logs=None):
        # At the end of each epoch, plot the gradients
        plt.figure(figsize=(10, 4))
        for i, g in enumerate(self.gradients[-1]):  # Get the last set of gradients
            plt.subplot(121)
            plt.hist(g.flatten(), bins=50)
            plt.xlabel('Gradients')
            plt.ylabel('Frequency')
            plt.title(f'Gradients Distribution of Layer {i + 1}')
            plt.subplot(122)
            plt.plot(g.flatten())
            plt.xlabel('Gradient index')
            plt.ylabel('Gradient value')
            plt.title(f'Gradients Line Plot of Layer {i + 1}')
        plt.tight_layout()
        plt.show()
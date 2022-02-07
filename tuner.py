from pyexpat import model
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras



class HpModel(kt.HyperModel):
    def __init__(self, model= None,loss = None, epochs =1):
        self.model = model
        self.loss = loss
        self.epochs = epochs
    def build(self, hp):
        """Builds a model."""
        return self.model(hp)

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # Convert the datasets to tf.data.Dataset.
        batch_size = hp.Int("batch_size", 500, 5000, step=500, default=2000)
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(
            batch_size
        )

        # Define the optimizer.
        optimizer = keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        )
        loss_fn = self.loss

        # The metric to track total loss.
        epoch_loss_val = keras.metrics.Mean()
        epoch_loss_train = keras.metrics.Mean()

        # Function to run the train step.
        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_train.update_state(loss)

        # Function to run the validation step.
        @tf.function
        def run_val_step(data, labels):
            logits = model(data)
            loss = loss_fn(labels, logits)
            # Update the metric.
            epoch_loss_val.update_state(loss)

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.model = model

        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # The custom training loop.
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")

            # Iterate the training data to run the training step.
            for data, labels in train_ds:
                run_train_step(data, labels)

            # Iterate the validation data to run the validation step.
            for data, labels in validation_data:
                run_val_step(data, labels)

            # Calling the callbacks after epoch.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_val.reset_states()
            epoch_loss_train.reset_states()
            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # Return the evaluation metric value.
        return best_epoch_loss
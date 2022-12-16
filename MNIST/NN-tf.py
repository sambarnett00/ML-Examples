import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def getTrainingDatasets():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = getTrainingDatasets()

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
        
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    model = MyModel()


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_obj(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_obj(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    EPOCHS = 5
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        
        print("------------------------------------------------")
        print(f"EPOCH={epoch+1}\nLoss={train_loss.result()}\nAccuracy={train_accuracy.result()*100}\nTest Loss={test_loss.result()}\nTest Accuracy={test_accuracy.result()*100}")

    model.save("v2Save")
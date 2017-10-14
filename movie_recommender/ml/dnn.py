from __future__ import print_function, division
import os
import tensorflow as tf
import numpy as np
import math

TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "training_data")
if not os.path.isdir(TRAINING_DATA_DIR):
    os.makedirs(TRAINING_DATA_DIR)
TMP_MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "model_tmp")
if not os.path.isdir(TMP_MODEL_DIR):
    os.makedirs(TMP_MODEL_DIR)


class DNN(object):
    def __init__(self, model_dir=None):
        self.classifier = None
        if model_dir is not None:
            print("Restoring classifier from {}".format(model_dir))
            self.classifier = tf.contrib.learn.DNNClassifier(model_dir=model_dir)

    def train(self, training_data, test_data):
        """

        :param training_data: Pandas dataframe for training
        :param test_data: Pandas dataframe for testing
        :return: The trained classifier.
        """
        print("Preparing input data...")
        training_csv_path, test_csv_path = self.make_tensorflow_training_files(training_data, test_data)
        target_column = training_data.columns.get_loc("rating")
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=training_csv_path,
            target_dtype=np.int,
            features_dtype=np.float,
            target_column=target_column)
        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=test_csv_path,
            target_dtype=np.int,
            features_dtype=np.float,
            target_column=target_column)
        feature_columns = [tf.feature_column.numeric_column("x", shape=[len(training_data.columns) - 1])]
        self.clean_directory(TMP_MODEL_DIR)
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 20, 10],
            n_classes=10,
            model_dir=TMP_MODEL_DIR
        )

        def get_test_inputs():
            x = tf.constant(test_set.data)
            y = tf.constant(test_set.target)
            return x, y

        def get_train_inputs():
            x = tf.constant(training_set.data)
            y = tf.constant(training_set.target)
            return x, y

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True
        )

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(test_set.data)},
            num_epochs=1,
            shuffle=False
        )

        print("Training classifier...")
        self.classifier.train(input_fn=train_input_fn, steps=2000)

        print("Testing classifier...")
        predictions = self.classifier.predict(input_fn=test_input_fn)
        predicted_classes = [int(p["classes"][0]) for p in predictions]
        actual_classes = test_data["rating"].tolist()
        mse = sum([(int(actual_classes[i])-int(predicted_classes[i]))**2 for i in range(len(actual_classes))])/len(actual_classes)
        print("\nTest Root MSE: {}\n".format(math.sqrt(mse)))

    @staticmethod
    def clean_directory(dir):
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            if os.path.isfile(path):
                os.unlink(path)

    @staticmethod
    def make_tensorflow_training_files(training_data, test_data):
        num_train_entries = training_data.shape[0]
        num_train_features = training_data.shape[1] - 1

        num_test_entries = test_data.shape[0]
        num_test_features = test_data.shape[1] - 1

        # The data frames are written as a temporary CSV file, as we still
        # need to modify the header row to include the number of rows and
        # columns present in the training and testing CSV files.

        training_data.to_csv(os.path.join(TRAINING_DATA_DIR, 'train_temp.csv'), index=False)
        test_data.to_csv(os.path.join(TRAINING_DATA_DIR, 'test_temp.csv'), index=False)

        # Append onto the header row the information about how many
        # columns and rows are in each file as TensorFlow requires.
        training_data_path = os.path.join(TRAINING_DATA_DIR, "movies_train.csv")
        test_data_path = os.path.join(TRAINING_DATA_DIR, "movies_test.csv")
        open(training_data_path, "w+").write(str(num_train_entries) +
                                              "," + str(num_train_features) +
                                              "," + open(os.path.join(TRAINING_DATA_DIR, "train_temp.csv")).read())

        open(test_data_path, "w+").write(str(num_test_entries) +
                                             "," + str(num_test_features) +
                                             "," + open(os.path.join(TRAINING_DATA_DIR, "test_temp.csv")).read())
        return training_data_path, test_data_path

    @staticmethod
    def make_valid_column_name(col):
        return col.replace(" ", "_")
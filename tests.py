from unittest import TestCase

from tensorflow.python.keras.losses import categorical_crossentropy

from genome_ac_gan_training import polyloss_ce, wasserstein_loss
from utils.util import *


class Test(TestCase):

    @staticmethod
    def change_values(input_array, max_value, min_value=0.0):
        tmp_array = input_array.copy()
        mask_zero = (tmp_array == 0)
        mask_one = (tmp_array == 1)
        # Generate random values between 0 and 0.3 for the zeros
        random_zero = np.random.uniform(min_value, max_value, size=mask_zero.sum())

        # Subtract random values between 0 and 0.3 from the ones
        random_one = np.random.uniform(min_value, max_value, size=mask_one.sum())
        # Subtract the random values from the ones
        tmp_array[mask_one] -= random_one

        # Add the random values to the zeros
        tmp_array[mask_zero] += random_zero
        return tmp_array

    def test_load_data(self):
        real_data = load_real_data(hapt_genotypes_path=REAL_10K_SNP_1000G_PATH,
                                   extra_data_path=REAL_EXTRA_DATA_PATH)
        real_data_population = real_data[get_relevant_columns(real_data, 'Population code')]
        real_data_population = filter_samples_by_minimum_examples(150, real_data_population,
                                                                  'Population code')
        real_data_super_population = real_data[get_relevant_columns(real_data,
                                                                    'Superpopulation code')]
        real_data_super_population = filter_samples_by_minimum_examples(10,
                                                                        real_data_super_population,
                                                                        'Superpopulation code')
        self.assertLess(len(real_data_population), len(real_data_super_population))


    def test_poly_loss(self):
        y_true = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
        y_pred = np.array([[0.15, 0.55, 0.15, 0.15], [0.3, 0.2, 0.1, 0.4]])

        # Convert test data to tensors
        y_true_tensor = tensorflow.convert_to_tensor(y_true, dtype=tensorflow.float32)
        y_pred_tensor = tensorflow.convert_to_tensor(y_pred, dtype=tensorflow.float32)

        # Calculate PolyLoss-CE score
        polyloss_ce_score = polyloss_ce(y_true_tensor, y_pred_tensor, epsilon=0.1, alpha=0.1)

        # Calculate the expected PolyLoss-CE score using the provided equations
        smooth_labels = y_true_tensor * (1 - 0.1) + 0.1 / y_true_tensor.shape[-1]
        one_minus_pt = tensorflow.reduce_sum(smooth_labels * (1 - y_pred_tensor), axis=-1)
        CE_loss = categorical_crossentropy(y_true_tensor, y_pred_tensor, label_smoothing=0.1)
        expected_polyloss_ce_score = CE_loss + 0.1 * one_minus_pt

        # Compare the calculated score with the expected score
        assert np.allclose(polyloss_ce_score.numpy(), expected_polyloss_ce_score.numpy())


    def test_loss(self):
        import tensorflow as tf
        y_true = np.array([-1,-1,-1])
        y_pred = np.array([2, -30, 20])
        min_max_value = 10.0

        expected_loss = np.mean(y_true * np.minimum(np.maximum(y_pred, -min_max_value), min_max_value))

        # Convert to TensorFlow tensors
        y_true_tf = tf.constant(y_true, dtype=tf.float32)
        y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
        min_max_value_tf = tf.constant(min_max_value, dtype=tf.float32)

        # Calculate the loss using the function
        loss = wasserstein_loss(y_true_tf, y_pred_tf, min_max_value_tf)

        # Check if the calculated loss matches the expected loss
        np.testing.assert_almost_equal(loss.numpy(), expected_loss, decimal=6)


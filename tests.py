from unittest import TestCase

from tensorflow.python.keras.losses import categorical_crossentropy

from genome_ac_gan_training import polyloss_ce
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
        real_data_population = real_data[get_relevant_columns(real_data, ['Population code'])]
        real_data_population = filter_samples_by_minimum_examples(150, real_data_population,
                                                                  'Population code')
        real_data_super_population = real_data[get_relevant_columns(real_data,
                                                                    ['Superpopulation code'])]
        real_data_super_population = filter_samples_by_minimum_examples(10,
                                                                        real_data_super_population,
                                                                        'Superpopulation code')
        self.assertLess(len(real_data_population), len(real_data_super_population))

    def test_polyloss_ce(self):
        epsilon = 0.1
        alpha = 0.2

        # Create example y_true and y_pred tensors
        y_true = tensorflow.constant([[0, 1, 0], [1, 0, 0]], dtype=tensorflow.float32)
        y_pred = tensorflow.constant([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]], dtype=tensorflow.float32)

        expected_loss = [0.62613606, 1.2523638]  # Updated expected Polyloss-CE scores for each sample

        # Calculate the actual Polyloss-CE scores
        actual_loss = polyloss_ce(y_true, y_pred, epsilon=epsilon, alpha=alpha)

        # Check if the actual Polyloss-CE scores match the expected scores
        for i in range(len(expected_loss)):
            self.assertAlmostEqual(actual_loss[i].numpy(), expected_loss[i], places=6)

    def test_get_smoothing_label_batch(self):
        # Create example y tensor
        y = tensorflow.constant([[0, 1, 0], [1, 0, 0]], dtype=tensorflow.float32)

        # Set the random seed for reproducibility
        tensorflow.random.set_seed(123)

        # Calculate the actual smoothed y tensor
        actual_smoothed_y = get_smoothing_label_batch(y)

        # Check if the actual smoothed y tensor matches the expected tensor
        for i in range(len(y)):
            for j in range(len(y[i])):
                self.assertGreater(actual_smoothed_y[i][j].numpy(), -0.5)
                self.assertLess(actual_smoothed_y[i][j].numpy(), 1)

from unittest import TestCase

import tensorflow
from sklearn.metrics import f1_score
from tensorflow.python.keras.losses import binary_crossentropy

from genome_ac_gan_training import f1_score_with_penalty, f1_loss_score
from utils import util
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

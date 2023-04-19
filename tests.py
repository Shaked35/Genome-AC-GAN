from unittest import TestCase

import tensorflow
from sklearn.metrics import f1_score
from tensorflow.python.keras.losses import binary_crossentropy

from gs_ac_gan_training import f1_score_with_penalty, f1_loss_score
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

    def test_f1_score(self):
        batch_size = 50
        num_classes = 5
        y_true = np.array(tensorflow.one_hot(
            tensorflow.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tensorflow.int32),
            depth=num_classes))
        very_good_pred = self.change_values(y_true, max_value=0.1)
        good_pred = self.change_values(y_true, max_value=0.4, min_value=0.2)
        bad_pred = self.change_values(y_true, max_value=0.7, min_value=0.5)
        very_bad_pred = self.change_values(y_true, max_value=0.95, min_value=0.7)
        best_score = f1_score_with_penalty(y_true, y_true)
        very_good_score = f1_score_with_penalty(y_true, very_good_pred)
        good_score = f1_score_with_penalty(y_true, good_pred)
        bad_score = f1_score_with_penalty(y_true, bad_pred)
        very_bad_score = f1_score_with_penalty(y_true, very_bad_pred)
        print(f"best_score:{best_score}, very_good_score:{very_good_score}, good_score:{good_score}, "
              f"bad_score:{bad_score}, very_bad_score:{very_bad_score}")

        self.assertLess(very_good_score, good_score)
        self.assertLess(good_score, bad_score)
        self.assertLess(bad_score, very_bad_score)
        self.assertLess(good_score, very_bad_score)

    def test_distance_f1_score(self):
        y_true = np.array([[0.0, 0.0, 0.0, 1.0]])
        pred1 = np.array([[0.1, 0.1, 0.1, 0.9]])
        pred2 = np.array([[0.1, 0.1, 0.9, 0.1]])
        pred3 = np.array([[0.1, 0.9, 0.1, 0.1]])
        pred4 = np.array([[0.9, 0.1, 0.1, 0.1]])
        util.class_weights = [1.123, 3.123, 2.433, 1.42]
        score1 = f1_score_with_penalty(y_true, pred1)
        score2 = f1_score_with_penalty(y_true, pred2)
        score3 = f1_score_with_penalty(y_true, pred3)
        score4 = f1_score_with_penalty(y_true, pred4)
        print(f"best_score:{score1}, score2:{score2}, score3:{score3}, score4:{score4}")

        self.assertLess(score1, score2)
        self.assertLess(score3, score4)

    def test_distance_bce_with_penalty(self):
        y_true = np.array([[0.0, 0.0, 0.0, 1.0]])
        pred1 = np.array([[0.1, 0.1, 0.1, 0.9]])
        pred2 = np.array([[0.1, 0.1, 0.9, 0.1]])
        pred3 = np.array([[0.1, 0.9, 0.1, 0.1]])
        pred4 = np.array([[0.9, 0.1, 0.1, 0.1]])
        util.class_weights = [0.5, 0.8, 1.123, 1.5]
        score1 = bce_with_penalty(y_true, pred1)
        score2 = bce_with_penalty(y_true, pred2)
        score3 = bce_with_penalty(y_true, pred3)
        score4 = bce_with_penalty(y_true, pred4)
        print(f"best_score:{score1}, score2:{score2}, score3:{score3}, score4:{score4}")

        self.assertLess(score1, score2)
        self.assertLess(score3, score4)

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

    def test_f1_score(self):
        y_true = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        y_pred1 = np.array([[0.1, 0.1, 0.9], [0.1, 0.1, 0.9], [0.1, 0.1, 0.9]])
        y_pred2 = np.array([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
        s_score1 = f1_score(y_true, tensorflow.round(y_pred1), average='micro')
        s_score2 = f1_score(y_true, tensorflow.round(y_pred2), average='micro')
        s_score3 = f1_loss_score(y_true, y_pred1)
        s_score4 = f1_loss_score(y_true, y_pred2)
        s_score5 = f1_loss_score(y_true, tensorflow.round(y_pred1))
        s_score6 = f1_loss_score(y_true, tensorflow.round(y_pred2))
        # s_score6 = binary_crossentropy(y_true, y_pred2)
        print(s_score1, s_score2, s_score3, s_score4, s_score5, s_score6)

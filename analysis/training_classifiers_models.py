from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, LeakyReLU
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from genome_ac_gan_training import polyloss_ce
from utils.util import *

init_gpus()

output_folder = "classifier_analysis"
train_set = '../resource/train_AFR_pop.csv'
test_set = '../resource/test_AFR_pop.csv'
experiment_results = '../fake_genotypes_sequences/new_sequences/sub_set_AFR_aug/genotypes.hapt'

# 17000: rf, knn
output_file = "classifiers_results1.csv"
target_column = 'Population code'

(x_train, y_train), class_id_to_counts, _, class_to_id = init_dataset(hapt_genotypes_path=train_set,
                                                                      target_column=target_column,
                                                                      with_extra_data=False)
id_to_class = {v: k for k, v in class_to_id.items()}

test_dataset = prepare_test_and_fake_dataset(experiment_results, test_path=test_set,
                                             target_column=target_column,
                                             class_to_id=class_to_id)


def calculate_accuracy(precision, recall):
    # Assuming precision and recall are in the range [0, 1]
    epsilon = 0.00001
    return (precision * recall) / ((precision + recall + epsilon) / 2)


def shuffle_test_dataset():
    indices = np.arange(test_dataset[0].shape[0])
    np.random.shuffle(indices)
    return (
        test_dataset[0][indices],
        np.array(test_dataset[1])[indices]
    )


y_train = np.argmax(y_train, axis=-1)
uniques, counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
percentages = counts / total_samples * 100
class_percentage_dict = dict(zip(uniques, percentages))
print(class_percentage_dict)

generated_samples = prepare_test_and_fake_dataset(experiment_results,
                                                  test_path=experiment_results,
                                                  from_generated=True,
                                                  class_to_id=class_to_id)
print(generated_samples[0].shape)
rows = []


def build_classifier(number_of_genotypes: int, alph: float):
    nn_classifier = Sequential([
        Dense(128, input_shape=(number_of_genotypes,),
              kernel_regularizer=regularizers.l2(0.0001)),

        LeakyReLU(alpha=alph),
        Dense(64, input_shape=(number_of_genotypes,),
              kernel_regularizer=regularizers.l2(0.0001)),
        LeakyReLU(alpha=alph),
        Dense(64, input_shape=(number_of_genotypes,),
              kernel_regularizer=regularizers.l2(0.0001)),

        LeakyReLU(alpha=alph),
        Dense(32, input_shape=(number_of_genotypes,),
              kernel_regularizer=regularizers.l2(0.0001)),
        LeakyReLU(alpha=alph),
        Dense(16, input_shape=(number_of_genotypes,),
              kernel_regularizer=regularizers.l2(0.0001)),
        LeakyReLU(alpha=alph),
        Dense(7, activation='softmax')
    ])

    nn_classifier.compile(optimizer=RMSprop(learning_rate=DEFAULT_DISCRIMINATOR_LEARNING_RATE),
                          loss=polyloss_ce,
                          metrics=['accuracy'])

    return nn_classifier


def concatenate_fake_data(percentage, generated_samples, x_train, y_train):
    if percentage == 0:
        train_dataset_with_generated_data = (x_train, y_train)
        num_samples = 0
    else:
        num_samples = int(percentage * len(generated_samples[0]))
        _, X_synthetic, _, Y_synthetic = train_test_split(generated_samples[0],
                                                          np.array(generated_samples[1]),
                                                          test_size=percentage,
                                                          random_state=None)
        print(Y_synthetic.shape)
        uniques, counts = np.unique(Y_synthetic, return_counts=True)
        total_samples = len(Y_synthetic)
        percentages = counts / total_samples * 100
        class_percentage_dict = dict(zip(uniques, percentages))
        print(class_percentage_dict)
        train_dataset_with_generated_data = (
            np.concatenate((x_train, X_synthetic), axis=0),
            np.concatenate((y_train, Y_synthetic), axis=0)
        )
    indices = np.arange(train_dataset_with_generated_data[0].shape[0])
    np.random.shuffle(indices)

    # Shuffle the dataset using the indices
    shuffled_dataset = (
        train_dataset_with_generated_data[0][indices],
        train_dataset_with_generated_data[1][indices]
    )
    return shuffled_dataset, num_samples


def init_models():
    knn = KNeighborsClassifier(n_neighbors=5, leaf_size=100)
    nn = build_classifier(10000, 0.01)
    rf = RandomForestClassifier()
    lgr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    svc = SVC(kernel='rbf')
    return {'KNN': knn, 'NN': nn, 'LGR': lgr}


scores = []
test_predictions = []
unique_models = 50
number_of_models = 0
test_dataset_shuffled = shuffle_test_dataset()

for i in range(unique_models):
    for synthetic_percentage in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        classifiers = init_models()
        start_time = datetime.now()
        for index, (model_name, clf) in enumerate(classifiers.items()):
            print(f"{i}: starting model {model_name} with percentage: {synthetic_percentage}")
            train_dataset_with_generated_data, number_of_synthetic_samples = concatenate_fake_data(
                percentage=synthetic_percentage,
                generated_samples=generated_samples,
                x_train=x_train, y_train=y_train)

            if model_name == 'NN':
                Y_train_encoded = tensorflow.one_hot(train_dataset_with_generated_data[1],
                                                     depth=train_dataset_with_generated_data[1].max() + 1)

                clf.fit(train_dataset_with_generated_data[0], Y_train_encoded,
                        batch_size=512, epochs=100, verbose=0)
                test_predictions = tensorflow.argmax(clf.predict(test_dataset_shuffled[0]), axis=1)

            else:
                clf.fit(train_dataset_with_generated_data[0], train_dataset_with_generated_data[1])
                test_predictions = clf.predict(test_dataset_shuffled[0])

            class_report = classification_report(test_dataset_shuffled[1], test_predictions, output_dict=True)

            class_accuracy = {}
            for class_label, metrics in class_report.items():
                if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                class_accuracy[class_label] = round(calculate_accuracy(metrics['precision'], metrics['recall']), 4)

            # Print accuracy by class
            output_row = {}
            for class_label, accuracy in class_accuracy.items():
                output_row[id_to_class[int(class_label)]] = accuracy
                print(f"Class {id_to_class[int(class_label)]}: {accuracy * 100:.2f}%")

            print(
                f"=======> {model_name} {synthetic_percentage}: f1_macro: {class_report['weighted avg']['f1-score']}, f1_weighted, {class_report['macro avg']['f1-score']}, accuracy score: {class_report['accuracy']} on synthetic_percentage: ")

            output_row.update({"synthetic_percentage": synthetic_percentage,
                               "samples_and_percentage": f"{number_of_synthetic_samples}\n{int(synthetic_percentage * 100)}%",
                               "model_name": model_name, "accuracy": class_report['accuracy'],
                               "f1_score": class_report['macro avg']['f1-score']})
            rows.append(output_row)
            number_of_models += 1
            if number_of_models % 10 == 0:
                print(f"finished train {number_of_models} models")
                pd.DataFrame(rows).to_csv(os.path.join("classifier_analysis", output_file))
    end_time = datetime.now()
    duration_minutes = (end_time - start_time).total_seconds() / 60

    print(f"finished model iteration in {duration_minutes} minutes")

pd.DataFrame(rows).to_csv(os.path.join("classifier_analysis", output_file))

import argparse
import json
import math
import os.path
from pathlib import Path

import pandas as pd
import tensorflow.python.keras.backend as K
from keras.layers import BatchNormalization
from sklearn.metrics import f1_score
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from utils.util import *

plt.switch_backend('agg')
class_weights = []


def save_mod(generator: Model, discriminator: Model, acgan: Model, experiment_results_path: str):
    discriminator.trainable = False
    acgan.save(os.path.join(experiment_results_path, "acgan"))
    acgan.save_weights(os.path.join(experiment_results_path, "acgan_weights"))
    discriminator.trainable = True
    generator.save(os.path.join(experiment_results_path, "generator"))
    generator.save_weights(os.path.join(experiment_results_path, "generator_weights"))
    discriminator.save(os.path.join(experiment_results_path, "discriminator"))
    discriminator.save_weights(os.path.join(experiment_results_path, "discriminator_weights"))


# Make generator
def build_generator(latent_dim: int, num_classes: int, number_of_genotypes: int, alph: float):
    generator = Sequential()
    generator.add(
        Dense(int(number_of_genotypes // 1.3), input_shape=(latent_dim + NUMBER_REPEAT_CLASS_VECTOR * num_classes,),
              kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(alpha=alph))
    generator.add(Dense(int(number_of_genotypes // 1.2), kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(alpha=alph))
    generator.add(Dense(int(number_of_genotypes // 1.1), kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(alpha=alph))
    generator.add(Dense(number_of_genotypes, activation='tanh'))

    # Generating the output image
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(num_classes,), dtype='float32')
    z = tensorflow.keras.layers.concatenate([label, label, label, label, noise, label, label, label, label])

    sequence = generator(z)

    return Model([noise, label], sequence)


# Make discriminator
def build_discriminator(number_of_genotypes: int, num_classes: int, alph: float, d_activation: str):
    # Define the sequence input

    discriminator = Sequential()
    discriminator.add(
        Dense(number_of_genotypes // 2, input_shape=(number_of_genotypes,), kernel_regularizer=regularizers.l2(0.0001)))

    discriminator.add(LeakyReLU(alpha=alph))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(number_of_genotypes // 3, kernel_regularizer=regularizers.l2(0.0001)))
    discriminator.add(LeakyReLU(alpha=alph))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(number_of_genotypes // 4, kernel_regularizer=regularizers.l2(0.0001)))
    discriminator.add(LeakyReLU(alpha=alph))
    discriminator.add(Dropout(0.1))
    sequence = Input(shape=(number_of_genotypes,), dtype='float32')

    # Extract features from images
    features = discriminator(sequence)

    # Building the output layer
    validity = Dense(1, activation=d_activation)(features)
    label = Dense(num_classes, activation="softmax")(features)

    return Model(sequence, [validity, label])


# Make GAN
def build_acgan(generator, discriminator):
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    acgan_output = discriminator(generator.output)
    acgan = Model(generator.input, acgan_output)
    return acgan


def train(batch_size: int, epochs: int, dataset: tuple, first_epoch: int, num_classes: int, latent_size: int,
          generator: Model, discriminator: Model, acgan: Model, save_number: int, class_id_to_counts: dict,
          experiment_results_path: str, id_to_class: dict, real_class_names: list, sequence_results_path: str,
          test_dataset):
    class_metric_results = None
    y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
    losses = []
    # Training iteration
    for e in range(first_epoch, epochs + 1):
        avg_d_loss, avg_g_loss = [], []
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(dataset).shuffle(
            dataset[0].shape[0]).batch(batch_size, drop_remainder=True)
        for x_batch_real, Y_batch_real in train_dataset:
            x_batch_real_with_noise = x_batch_real - np.random.uniform(0, 0.1, size=(
                x_batch_real.shape[0], x_batch_real.shape[1]))
            latent_samples = np.random.normal(loc=0, scale=1,
                                              size=(batch_size, latent_size))  # create noise to be fed to generator
            fake_labels_batch = tensorflow.one_hot(
                tensorflow.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tensorflow.int32),
                depth=num_classes)

            X_batch_fake = generator.predict_on_batch([latent_samples, fake_labels_batch])

            # train discriminator, notice that noise is added to real labels
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(x_batch_real_with_noise, [y_real - np.random.uniform(0, 0.1, size=(
                y_real.shape[0], y_real.shape[1])), Y_batch_real])

            d_loss += discriminator.train_on_batch(X_batch_fake, [y_fake, fake_labels_batch])
            d_loss = (d_loss[0] + d_loss[5]) / 2
            # make discriminator non-trainable and train gan
            discriminator.trainable = False
            g_loss = acgan.train_on_batch([latent_samples, fake_labels_batch],
                                          [y_real, fake_labels_batch])
            g_loss = g_loss[0]
            avg_d_loss.append(d_loss)
            avg_g_loss.append(g_loss)

        losses.append((np.average(avg_d_loss), np.average(avg_g_loss)))

        # real_fake, class_predictions = discriminator.predict_on_batch([x_batch_real_with_noise])
        # y_pred = tensorflow.argmax(class_predictions, axis=1)
        # uniques_test, counts_test = np.unique(y_pred, return_counts=True)
        # class_id_to_counts_test = dict(zip(uniques_test, counts_test))
        # print("class_id_to_counts: ", class_id_to_counts_test)

        print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (
            e, epochs, np.average(avg_d_loss), np.average(avg_g_loss)))
        if e % save_number == 0 or e == epochs:
            class_metric_results = save_discriminator_class_pred(discriminator, test_dataset, experiment_results_path,
                                                                 id_to_class, class_metric_results, e)
            # Save models
            save_mod(acgan=acgan, generator=generator, discriminator=discriminator,
                     experiment_results_path=experiment_results_path)

            plot_pca_comparisons(generator=generator, epoch_number=e,
                                 losses=losses, class_id_to_counts=class_id_to_counts,
                                 experiment_results_path=experiment_results_path,
                                 latent_size=latent_size, num_classes=num_classes, dataset=dataset,
                                 id_to_class=id_to_class, real_class_names=real_class_names,
                                 sequence_results_path=sequence_results_path)


def save_discriminator_class_pred(discriminator, test_dataset, experiment_results_path, id_to_class,
                                  class_metric_results, epoch):
    shuffled_dataset = tensorflow.data.Dataset.from_tensor_slices(test_dataset).shuffle(
        test_dataset[0].shape[0]).batch(test_dataset[0].shape[0], drop_remainder=True)
    for x_batch_real, y_batch_real in shuffled_dataset:
        r_f, class_predictions = discriminator.predict_on_batch(x_batch_real)
        y_pred = tensorflow.argmax(class_predictions, axis=1)
        class_classifier_results = pd.DataFrame({"class_pred": np.array(y_pred), "class_real": np.array(y_batch_real),
                                                 "real_fake_pred": r_f.flatten()})
        class_classifier_results["class_name_real"] = class_classifier_results["class_real"].replace(id_to_class)
        class_classifier_results["class_name_pred"] = class_classifier_results["class_pred"].replace(id_to_class)
        class_classifier_results.to_csv(
            os.path.join(experiment_results_path, "discriminator_pred_on_test.csv"), index=False)
        class_names = list(set(id_to_class.values()))
        plot_confusion_matrix(class_classifier_results["class_real"],
                              class_classifier_results["class_pred"], class_names,
                              experiment_results_path)

        row_results = compute_metrics(class_classifier_results["class_real"],
                                      class_classifier_results["class_pred"])
        row_results["epoch"] = epoch
        row_results = pd.DataFrame([row_results])
        if class_metric_results is None:
            class_metric_results = pd.DataFrame(row_results)
        else:
            class_metric_results = pd.concat([class_metric_results, row_results])
        class_metric_results.to_csv(os.path.join(experiment_results_path, DISCRIMINATOR_METRICS_FILE),
                                    index=False)
        return class_metric_results


def plot_pca_comparisons(generator: Model, epoch_number: int, losses: list,
                         class_id_to_counts: dict, num_classes: int, latent_size: int, experiment_results_path: str,
                         dataset: tuple, id_to_class: dict, real_class_names: list, sequence_results_path: str):
    # Create AGs
    generated_genomes_total = []
    for class_id, number_of_sequences in class_id_to_counts.items():
        fake_labels_batch = tensorflow.one_hot(np.full(shape=(number_of_sequences,), fill_value=class_id),
                                               depth=num_classes)
        latent_samples = np.random.normal(loc=0, scale=1, size=(number_of_sequences, latent_size))
        generated_genomes = generator.predict([latent_samples, fake_labels_batch])
        generated_genomes[generated_genomes < 0] = 0
        generated_genomes = np.rint(generated_genomes)
        tmp_generated_genomes_df = pd.DataFrame(generated_genomes)
        tmp_generated_genomes_df = tmp_generated_genomes_df.astype(int)
        tmp_generated_genomes_df.insert(loc=0, column='Type', value=f"Fake_{id_to_class[class_id]}")
        generated_genomes_total.append(tmp_generated_genomes_df)
    generated_genomes_df = pd.concat(generated_genomes_total)
    generated_genomes_df.to_csv(os.path.join(sequence_results_path, "genotypes.hapt"), sep=" ", header=False,
                                index=False)
    plt.rcParams['figure.max_open_warning'] = 50  # set the max number of figures before the warning is triggered to 50

    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(experiment_results_path, str(epoch_number) + '_loss.pdf'), format='pdf')
    # Make PCA
    real_sequences = dataset[0].copy()
    real_sequences[real_sequences < 0] = 0
    real_sequences = np.rint(real_sequences)
    real_sequences = pd.DataFrame(real_sequences)
    real_sequences['Type'] = real_class_names
    df_all_pca = pd.concat([real_sequences, generated_genomes_df])
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(df_all_pca.drop(['Type'], axis=1))
    PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
    PCs_df['Class'] = list(df_all_pca['Type'])
    # Define the colors for real and fake points
    real_color = 'blue'
    fake_color = 'red'
    # Define the populations
    populations = list(id_to_class.values())
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(nrows=math.ceil((len(populations) + 1) / 2), ncols=2,
                            figsize=(16, len(populations) * 4))
    # Loop over each population and plot the real and fake points separately
    column_mod = 2
    row = 0
    alpha_color = 0.3
    for i, pop in enumerate(populations):
        # Get the indices of the real and fake points for this population
        real_idx = (PCs_df['Class'] == f'Real_{pop}')
        fake_idx = (PCs_df['Class'] == f'Fake_{pop}')

        # Get the PCA values for the real and fake points
        real_pca = PCs_df.loc[real_idx, ['PC1', 'PC2']]
        fake_pca = PCs_df.loc[fake_idx, ['PC1', 'PC2']]

        # Plot the real and fake points in separate subplots
        axs[row, column_mod % 2].scatter(real_pca['PC1'], real_pca['PC2'], c=real_color, label='Real',
                                         alpha=alpha_color)
        axs[row, column_mod % 2].scatter(fake_pca['PC1'], fake_pca['PC2'], c=fake_color, label='Fake',
                                         alpha=alpha_color)
        axs[row, column_mod % 2].set_xlabel('PC1')
        axs[row, column_mod % 2].set_ylabel('PC2')
        axs[row, column_mod % 2].set_title(f'{pop} - Real vs Fake')
        axs[row, column_mod % 2].legend()
        column_mod += 1
        row += 1 if column_mod % 2 == 0 else 0
    # Get the indices of the real and fake points for this population
    real_idx = PCs_df.index[PCs_df['Class'].str.contains('Real')]
    fake_idx = PCs_df.index[PCs_df['Class'].str.contains('Fake')]
    # Get the PCA values for the real and fake points
    real_pca = PCs_df.loc[real_idx, ['PC1', 'PC2']]
    fake_pca = PCs_df.loc[fake_idx, ['PC1', 'PC2']]
    # Plot the real and fake points together in another subplot
    axs[row, 1].scatter(real_pca['PC1'], real_pca['PC2'], c=real_color, label='Real',
                        alpha=alpha_color)
    axs[row, 1].scatter(fake_pca['PC1'], fake_pca['PC2'], c=fake_color, label='Fake',
                        alpha=alpha_color)
    axs[row, 1].set_xlabel('PC1')
    axs[row, 1].set_ylabel('PC2')
    axs[row, 1].set_title(f'All - Real vs Fake')
    axs[row, 1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_results_path, str(epoch_number) + '_pca.pdf'), format='pdf')
    plt.close()
    plt.switch_backend('agg')


def f1_loss_for_labels(y_true, y_pred):
    return f1_loss_for_one_hot(y_true, y_pred, average='micro', rounded=True)


def f1_loss_for_one_hot(y_true, y_pred, average, rounded=False):
    if rounded:
        tmp_y_pred = tensorflow.round(y_pred)
    else:
        tmp_y_pred = y_pred
    precision, recall = calculate_precision_recall(tmp_y_pred, y_true)
    f1 = 2 * precision * recall / (precision + recall + tensorflow.keras.backend.epsilon())

    if average == 'micro':
        tp = tensorflow.reduce_sum(y_true * tmp_y_pred)
        fp = tensorflow.reduce_sum(tmp_y_pred) - tp
        fn = tensorflow.reduce_sum(y_true) - tp
        f1_micro = (2 * tp) / (2 * tp + fp + fn)
        return 1 - f1_micro
    elif average == 'macro':
        return 1 - f1
    elif average == 'weighted':
        weights = tensorflow.reduce_sum(y_true, axis=0)
        weights /= tensorflow.reduce_sum(weights)
        return 1 - tensorflow.reduce_sum(f1 * tensorflow.cast(weights, dtype=tensorflow.float32))
    elif average == 'samples':
        tp = tensorflow.reduce_sum(y_true * tmp_y_pred, axis=1)
        fp = tensorflow.reduce_sum(tmp_y_pred, axis=1) - tp
        fn = tensorflow.reduce_sum(y_true, axis=1) - tp
        f1_samples = (2 * tp) / (2 * tp + fp + fn)
        return 1 - tensorflow.reduce_mean(f1_samples)
    return None


def f1_loss_score_macro(y_true, y_pred):
    return f1_loss_for_one_hot(y_true, y_pred, average='macro')


def f1_loss_score_micro(y_true, y_pred):
    return f1_loss_for_one_hot(y_true, y_pred, average='micro')


def f1_loss_score_weighted(y_true, y_pred):
    return f1_loss_for_one_hot(y_true, y_pred, average='weighted')


def f1_loss_score_samples(y_true, y_pred):
    return f1_loss_for_one_hot(y_true, y_pred, average='samples')


def calculate_precision_recall(y_pred, y_real):
    true_positives = tensorflow.reduce_sum(tensorflow.cast(y_real * y_pred, dtype=tensorflow.float32))
    predicted_positives = tensorflow.reduce_sum(tensorflow.cast(y_pred, dtype=tensorflow.float32))
    actual_positives = tensorflow.reduce_sum(tensorflow.cast(y_real, dtype=tensorflow.float32))
    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    return precision, recall


def calculate_distance_penalty(y_pred, y_real):
    true_class = tensorflow.argmax(y_real, axis=1)
    fake_class = tensorflow.argmax(y_pred, axis=1)
    distance_penalty = tensorflow.reduce_mean(
        tensorflow.cast((tensorflow.abs(true_class - fake_class)) / (y_real.shape[1] * 2),
                        dtype=tensorflow.float32))
    return distance_penalty


def prepare_test_data(experiment_results, test_path="resource/test_0.2_super_pop.csv"):
    target_column = "Superpopulation code"

    with open(os.path.join(experiment_results, "class_id_map.json"), 'r') as file:
        json_data = file.read()

    class_to_id = json.loads(json_data)
    test_set = pd.read_csv(test_path)
    relevant_columns = get_relevant_columns(test_set, [SAMPLE_COLUMN_NAME, target_column])
    test_set = filter_samples_by_minimum_examples(10, test_set, target_column)
    test_set = test_set.sample(frac=1, random_state=np.random.RandomState(seed=42))
    _, _, y_real = extract_y_column(class_to_id, test_set, target_column)
    y_real = tensorflow.argmax(y_real, axis=1)
    x_values = extract_x_values(test_set, relevant_columns, target_column)
    return x_values, y_real


# define a function to calculate class weights based on inverse frequency
def set_class_weights(y):
    n_samples = len(y)
    n_classes = y.shape[1]
    class_counts = np.sum(y, axis=0)
    for i in range(n_classes):
        class_weights.append((n_samples - class_counts[i]) / class_counts[i])


def main(hapt_genotypes_path: str, extra_data_path: str, experiment_results_path: str, latent_size: int, alph: float,
         g_learn: float, d_learn: float, epochs: int, batch_size: int, class_loss_weights: float, save_number: int,
         minimum_samples: int, first_epoch: int, target_column: str, sequence_results_path: str, d_activation: str,
         class_loss_function: str, validation_loss_function: str, without_extra_data: bool):
    K.clear_session()
    init_gpus()
    target_column = " ".join(target_column.split("_"))
    dataset, class_id_to_counts, num_classes, class_to_id = init_dataset(hapt_genotypes_path=hapt_genotypes_path,
                                                                         extra_data_path=extra_data_path,
                                                                         target_column=target_column,
                                                                         minimum_samples=minimum_samples,
                                                                         without_extra_data=without_extra_data)
    set_class_weights(dataset[1])
    # save class id map
    with open(os.path.join(experiment_results, 'class_id_map.json'), 'w') as f:
        json.dump(class_to_id, f)
    number_of_genotypes = dataset[0].shape[1]
    id_to_class = reverse_dict(class_to_id)
    print(f"dataset shapes: {dataset[0].shape, dataset[1].shape}")
    print(f"classes: {num_classes}, class_id_to_counts: {class_id_to_counts}")
    real_class_names = pd.DataFrame(np.argmax(dataset[1], 1))
    real_class_names = real_class_names[0].replace(id_to_class)
    real_class_names = 'Real_' + real_class_names
    real_class_names = list(real_class_names)
    generator = build_generator(latent_dim=latent_size, num_classes=num_classes,
                                number_of_genotypes=number_of_genotypes, alph=alph)

    generator.compile(metrics=['accuracy'])
    discriminator = build_discriminator(number_of_genotypes=number_of_genotypes, num_classes=num_classes, alph=alph,
                                        d_activation=d_activation)

    if class_loss_function == "f1_loss_score_macro":
        class_loss_function = f1_loss_score_macro
    if class_loss_function == "f1_loss_score_micro":
        class_loss_function = f1_loss_score_micro
    if class_loss_function == "f1_loss_score_weighted":
        class_loss_function = f1_loss_score_weighted
    if class_loss_function == "f1_loss_score_samples":
        class_loss_function = f1_loss_score_samples
    if class_loss_function == "f1_loss_for_labels":
        class_loss_function = f1_loss_for_labels

    discriminator.compile(optimizer=RMSprop(lr=d_learn), loss=[validation_loss_function, class_loss_function],
                          loss_weights=[1, class_loss_weights], metrics=['accuracy'])
    # Set discriminator to non-trainable
    discriminator.trainable = False
    acgan = build_acgan(generator, discriminator)

    acgan.compile(optimizer=RMSprop(lr=g_learn), loss=[validation_loss_function, class_loss_function],
                  loss_weights=[1, class_loss_weights],
                  metrics=['accuracy'])

    test_dataset = prepare_test_data(experiment_results)
    train(batch_size=batch_size, epochs=epochs, dataset=dataset, first_epoch=first_epoch, num_classes=num_classes,
          latent_size=latent_size, generator=generator, discriminator=discriminator, acgan=acgan,
          save_number=save_number, class_id_to_counts=class_id_to_counts,
          experiment_results_path=experiment_results_path, id_to_class=id_to_class, real_class_names=real_class_names,
          sequence_results_path=sequence_results_path, test_dataset=test_dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='GS-AC-GAN training parser')
    parser.add_argument('--hapt_genotypes_path', type=str, default=REAL_10K_SNP_1000G_PATH,
                        help='path to real input hapt file')
    parser.add_argument('--experiment_name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help='experiment name')
    parser.add_argument('--extra_data_path', type=str, default=REAL_EXTRA_DATA_PATH,
                        help='path to real extra data with classes file')
    parser.add_argument('--results_folder', type=str, default=DEFAULT_RESULTS_FOLDER,
                        help='where put the output results')
    parser.add_argument('--latent_size', type=int, default=DEFAULT_LATENT_SIZE,
                        help='input noise latent size')
    parser.add_argument('--alph', type=float, default=DEFAULT_ALPH, help='alpha value for LeakyReLU')
    parser.add_argument('--g_learn', type=float, default=DEFAULT_GENERATOR_LEARNING_RATE,
                        help='generator learning rate')
    parser.add_argument('--d_learn', type=float, default=DEFAULT_DISCRIMINATOR_LEARNING_RATE,
                        help='discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='initial batch size')
    parser.add_argument('--class_loss_weights', type=float, default=DEFAULT_CLASS_LOSS_WEIGHTS,
                        help='what is the weight to calculate the loss score for discriminator classes and generator classes')
    parser.add_argument('--save_number', type=int, default=DEFAULT_SAVE_NUMBER,
                        help='how often to save the results')
    parser.add_argument('--minimum_samples', type=int, default=DEFAULT_MINIMUM_SAMPLES,
                        help='what is the minimum samples that we find to include the class in the training process')
    parser.add_argument('--first_epoch', type=int, default=DEFAULT_FIRST_EPOCH,
                        help='what is the first epoch number')
    parser.add_argument('--target_column', type=str, default=DEFAULT_TARGET_COLUMN,
                        help='class column name', choices=['Population_code', 'Population_name',
                                                           'Superpopulation_code', 'Superpopulation_name'])
    parser.add_argument('--d_activation', type=str, default=DEFAULT_DISCRIMINATOR_ACTIVATION,
                        help='discriminator validation activation function (real/fake)')
    parser.add_argument('--class_loss_function', type=str, default=DEFAULT_CLASS_LOSS_FUNCTION,
                        help='loss function between different classes')
    parser.add_argument('--validation_loss_function', type=str, default=DEFAULT_VALIDATION_LOSS_FUNCTION,
                        help='loss function between different real/fake')
    parser.add_argument('--without_extra_data', type=bool, default=False,
                        help="don't need to load extra data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiment_results = os.path.join(args.results_folder, args.experiment_name)
    sequence_results = os.path.join(SEQUENCE_RESULTS_PATH, args.experiment_name)
    Path(experiment_results).mkdir(parents=True, exist_ok=True)
    Path(sequence_results).mkdir(parents=True, exist_ok=True)

    # save the args as a JSON file
    with open(os.path.join(experiment_results, 'experiment_args.json'), 'w') as f:
        json.dump(vars(args), f)

    main(hapt_genotypes_path=args.hapt_genotypes_path,
         extra_data_path=args.extra_data_path,
         experiment_results_path=experiment_results,
         latent_size=args.latent_size,
         alph=args.alph,
         g_learn=args.g_learn,
         d_learn=args.d_learn,
         epochs=args.epochs,
         batch_size=args.batch_size,
         class_loss_weights=args.class_loss_weights,
         save_number=args.save_number,
         minimum_samples=args.minimum_samples,
         first_epoch=args.first_epoch,
         target_column=args.target_column,
         sequence_results_path=sequence_results,
         d_activation=args.d_activation,
         class_loss_function=args.class_loss_function,
         validation_loss_function=args.validation_loss_function,
         without_extra_data=args.without_extra_data)

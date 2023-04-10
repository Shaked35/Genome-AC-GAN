import argparse
import json
import math
from pathlib import Path

import tensorflow.python.keras.backend as K
from keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from utils.util import *


plt.switch_backend('agg')


def save_mod(generator: Model, discriminator: Model, acgan: Model, experiment_results_path: str):
    discriminator.trainable = False
    save_model(acgan, os.path.join(experiment_results_path, "acgan"))
    discriminator.trainable = True
    save_model(generator, os.path.join(experiment_results_path, "generator"))
    save_model(discriminator, os.path.join(experiment_results_path, "discriminator"))


# Make generator
def build_generator(latent_dim: int, num_classes: int, number_of_genotypes: int, alph: float):
    generator = Sequential()
    generator.add(
        Dense(int(number_of_genotypes // 1.2), input_shape=(latent_dim + NUMBER_REPEAT_CLASS_VECTOR * num_classes,),
              kernel_regularizer=regularizers.l2(0.0001)))
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
    discriminator.add(Dense(number_of_genotypes // 3, kernel_regularizer=regularizers.l2(0.0001)))
    discriminator.add(LeakyReLU(alpha=alph))

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
          experiment_results_path: str, id_to_class: dict, real_class_names: list, sequence_results_path: str):
    y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
    losses = []
    d_loss, g_loss = 0, 0
    # Training iteration
    for e in range(first_epoch, epochs + 1):
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(dataset).shuffle(
            dataset[0].shape[0]).batch(batch_size, drop_remainder=True)
        for X_batch_real, Y_batch_real in train_dataset:
            latent_samples = np.random.normal(loc=0, scale=1,
                                              size=(batch_size, latent_size))  # create noise to be fed to generator
            fake_labels_batch = tensorflow.one_hot(
                tensorflow.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tensorflow.int32),
                depth=num_classes)

            X_batch_fake = generator.predict_on_batch(
                [latent_samples, fake_labels_batch])  # create batch from generator using noise as input

            # train discriminator, notice that noise is added to real labels
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X_batch_real, [y_real - np.random.uniform(0, 0.1, size=(
                y_real.shape[0], y_real.shape[1])), np.array(Y_batch_real - np.random.uniform(0, 0.1, size=(
                Y_batch_real.shape[0], Y_batch_real.shape[1])))])
            d_loss += discriminator.train_on_batch(X_batch_fake,
                                                   [y_fake,
                                                    np.array(fake_labels_batch - np.random.uniform(0, 0.1, size=(
                                                        fake_labels_batch.shape[0], fake_labels_batch.shape[1])))])
            d_loss = (d_loss[0] + d_loss[3]) / 2
            # make discriminator non-trainable and train gan
            discriminator.trainable = False
            g_loss = acgan.train_on_batch([latent_samples, fake_labels_batch],
                                          [y_real, np.array(fake_labels_batch - np.random.uniform(0, 0.1, size=(
                                              fake_labels_batch.shape[0], fake_labels_batch.shape[1])))])
            g_loss = g_loss[0]

        losses.append((d_loss, g_loss))
        print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (e, epochs, d_loss, g_loss))
        if e % save_number == 0 or e == epochs:
            plot_pca_comparisons(discriminator=discriminator, acgan=acgan, generator=generator, epoch_number=e,
                                 losses=losses, class_id_to_counts=class_id_to_counts,
                                 experiment_results_path=experiment_results_path,
                                 latent_size=latent_size, num_classes=num_classes, dataset=dataset,
                                 id_to_class=id_to_class, real_class_names=real_class_names,
                                 sequence_results_path=sequence_results_path)


def plot_pca_comparisons(discriminator: Model, acgan: Model, generator: Model, epoch_number: int, losses: list,
                         class_id_to_counts: dict, num_classes: int, latent_size: int, experiment_results_path: str,
                         dataset: tuple, id_to_class: dict, real_class_names: list, sequence_results_path: str):
    # Save models
    save_mod(acgan, generator, discriminator, experiment_results_path)
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
    fig, ax = plt.subplots()
    plt.plot(np.array([losses]).T[0], label='Discriminator')
    plt.plot(np.array([losses]).T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    fig.savefig(os.path.join(experiment_results_path, str(epoch_number) + '_loss.pdf'), format='pdf')
    # Make PCA
    df_pca = dataset[0].copy()
    df_pca[df_pca < 0] = 0
    df_pca = np.rint(df_pca)
    df_pca = pd.DataFrame(df_pca)
    df_pca['Type'] = real_class_names
    df_all_pca = pd.concat([df_pca, generated_genomes_df])
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
    fig, axs = plt.subplots(nrows=math.ceil(len(populations) / 2), ncols=2, figsize=(14, 20))
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


def wasserstein_loss(y_true, y_pred):
    return tensorflow.reduce_mean((y_true * y_pred))


def f1_score_with_penalty(y_real, y_pred):
    true_positives = tensorflow.reduce_sum(tensorflow.cast(y_real * y_pred, dtype=tensorflow.float32))
    predicted_positives = tensorflow.reduce_sum(tensorflow.cast(y_pred, dtype=tensorflow.float32))
    actual_positives = tensorflow.reduce_sum(tensorflow.cast(y_real, dtype=tensorflow.float32))
    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    distance_penalty = calculate_distance_penalty(y_pred, y_real)
    return 1 - 2 * ((precision * recall) / (precision + recall + 1e-16)) + distance_penalty


def calculate_distance_penalty(y_pred, y_real):
    true_class = tensorflow.argmax(y_real)
    fake_class = tensorflow.argmax(y_pred)
    distance_penalty = tensorflow.reduce_mean(
        tensorflow.cast((tensorflow.abs(true_class - fake_class)) / (y_real.shape[0]),
                        dtype=tensorflow.float32))
    return distance_penalty


def wasserstein_class_loss(y_true, y_pred):
    # Convert one-hot encoded y_true to integers
    y_true_int = tensorflow.cast(K.argmax(y_true, axis=-1), tensorflow.float32)

    # Extract the relevant probabilities from y_pred
    y_pred_relevant = K.sum(y_true * y_pred, axis=-1)

    # Compute the Wasserstein distance
    return K.mean(y_true_int * y_pred_relevant)


def main(hapt_genotypes_path: str, extra_data_path: str, experiment_results_path: str, latent_size: int, alph: float,
         g_learn: float, d_learn: float, epochs: int, batch_size: int, class_loss_weights: float, save_number: int,
         minimum_samples: int, first_epoch: int, target_column: str, sequence_results_path: str, d_activation: str,
         class_loss_function: str, validation_loss_function: str):
    K.clear_session()
    init_gpus()
    dataset, class_id_to_counts, num_classes, class_to_id = init_dataset(hapt_genotypes_path=hapt_genotypes_path,
                                                                         extra_data_path=extra_data_path,
                                                                         target_column=target_column,
                                                                         minimum_samples=minimum_samples)
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

    class_loss_function = f1_score_with_penalty if class_loss_function == 'f1_score_with_penalty' else class_loss_function
    discriminator.compile(optimizer=RMSprop(lr=d_learn), loss=[validation_loss_function, class_loss_function],
                          loss_weights=[1, class_loss_weights], metrics=['accuracy'])
    # Set discriminator to non-trainable
    discriminator.trainable = False
    acgan = build_acgan(generator, discriminator)

    acgan.compile(optimizer=RMSprop(lr=g_learn), loss=[validation_loss_function, class_loss_function],
                  loss_weights=[1, class_loss_weights],
                  metrics=['accuracy'])
    train(batch_size=batch_size, epochs=epochs, dataset=dataset, first_epoch=first_epoch, num_classes=num_classes,
          latent_size=latent_size, generator=generator, discriminator=discriminator, acgan=acgan,
          save_number=save_number, class_id_to_counts=class_id_to_counts,
          experiment_results_path=experiment_results_path, id_to_class=id_to_class, real_class_names=real_class_names,
          sequence_results_path=sequence_results_path)


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
                        help='class column name', choices=['Population code', 'Population name',
                                                           'Superpopulation code', 'Superpopulation name'])
    parser.add_argument('--d_activation', type=str, default=DEFAULT_DISCRIMINATOR_ACTIVATION,
                        help='discriminator validation activation function (real/fake)')
    parser.add_argument('--class_loss_function', type=str, default=DEFAULT_CLASS_LOSS_FUNCTION,
                        help='loss function between different classes')
    parser.add_argument('--validation_loss_function', type=str, default=DEFAULT_VALIDATION_LOSS_FUNCTION,
                        help='loss function between different real/fake')
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
         validation_loss_function=args.validation_loss_function)

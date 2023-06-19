import argparse
import os.path
from pathlib import Path

import tensorflow.python.keras.backend as K
import tensorflow.python.ops.numpy_ops
from keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.python.keras.metrics import CategoricalAccuracy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from utils.util import *

plt.switch_backend('agg')


# Make Generator Model
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


# Make Discriminator Model
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


# Make AC-GAN Model
def build_acgan(generator, discriminator):
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    acgan_output = discriminator(generator.output)
    acgan = Model(generator.input, acgan_output)
    return acgan


def train(batch_size: int, epochs: int, dataset: tuple, num_classes: int, latent_size: int,
          generator: Model, discriminator: Model, acgan: Model, save_number: int, class_id_to_counts: dict,
          experiment_results_path: str, id_to_class: dict, real_class_names: list, sequence_results_path: str,
          test_dataset):
    """
    genome-ac-gan training process
    :param batch_size: int batch size training
    :param epochs: int number of epochs
    :param dataset: tuple of 2 np.arrays (x, y) sequences of genotypes and labels
    :param num_classes: number of classes in Y_true
    :param latent_size: input latent size for noise(z)
    :param generator: generator model
    :param discriminator: discriminator model
    :param acgan: acgan model
    :param save_number: how many epochs between saving the temp results
    :param class_id_to_counts: map of class translate to id
    :param experiment_results_path: output folder results path
    :param id_to_class: translation of id to class name
    :param real_class_names: all Y_true labels names
    :param sequence_results_path: path to folder that will contain the synthetic sequences
    :param test_dataset: dataset to evaluate the classifier
    """
    class_metric_results = None
    y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
    losses = []
    # Training iteration
    for e in range(1, epochs + 1):
        avg_d_loss, avg_g_loss = [], []
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(dataset).shuffle(
            dataset[0].shape[0]).batch(batch_size, drop_remainder=True)
        for x_batch_real, Y_batch_real in train_dataset:
            x_batch_real_with_noise = x_batch_real - np.random.uniform(0, 0.1, size=(
                x_batch_real.shape[0], x_batch_real.shape[1]))
            latent_samples = np.random.normal(loc=0, scale=1,
                                              size=(batch_size, latent_size))  # create noise to be input to generator
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
            discriminator.trainable = False
            # finished training discriminator, now train generator
            g_loss = acgan.train_on_batch([latent_samples, fake_labels_batch],
                                          [y_real, fake_labels_batch])
            g_loss = g_loss[0]
            avg_d_loss.append(d_loss)
            avg_g_loss.append(g_loss)

        losses.append((np.average(avg_d_loss), np.average(avg_g_loss)))

        print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (
            e, epochs, np.average(avg_d_loss), np.average(avg_g_loss)))
        if e % save_number == 0 or e == epochs:
            if test_dataset is not None:
                # evaluate the classifier
                class_metric_results = save_discriminator_class_pred(discriminator, test_dataset,
                                                                     experiment_results_path, id_to_class,
                                                                     class_metric_results, e)

            # save all models
            save_models(acgan=acgan, generator=generator, discriminator=discriminator,
                        experiment_results_path=experiment_results_path)

            # plot PCA for all population and for each label
            plot_pca_comparisons(generator=generator, epoch_number=e,
                                 class_id_to_counts=class_id_to_counts,
                                 experiment_results_path=experiment_results_path,
                                 latent_size=latent_size, num_classes=num_classes, dataset=dataset,
                                 id_to_class=id_to_class, real_class_names=real_class_names,
                                 sequence_results_path=sequence_results_path)


def polyloss_ce(y_true, y_pred, epsilon=DEFAULT_EPSILON_LCE, alpha=DEFAULT_ALPH_LCE):
    """
    Polyloss-CE classification loss function that describes in the article:
    "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"
    https://arxiv.org/pdf/2204.12511.pdf
    :param y_true: sequences input y true labels
    :param y_pred: sequences input y predictions labels
    :param epsilon: epsilon >=-1, penalty weight
    :param alpha: 1>=alpha>=0 smooth labels percentages
    :return: Polyloss-CE score
    """
    num_classes = y_true.get_shape().as_list()[-1]
    smooth_labels = y_true * (1 - alpha) + alpha / num_classes
    one_minus_pt = tensorflow.reduce_sum(smooth_labels * (1 - y_pred), axis=-1)
    CE_loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=alpha,
                                                              reduction='none')
    CE = CE_loss(y_true, y_pred)
    Poly1 = CE + epsilon * one_minus_pt
    return Poly1


def main(hapt_genotypes_path: str, extra_data_path: str, experiment_results_path: str, latent_size: int, alph: float,
         g_learn: float, d_learn: float, epochs: int, batch_size: int, class_loss_weights: float, save_number: int,
         minimum_samples: int, target_column: str, sequence_results_path: str, d_activation: str,
         class_loss_function: str, validation_loss_function: str, with_extra_data: bool,
         test_discriminator_classifier: bool, required_populations: list[str]):
    K.clear_session()
    init_gpus()
    target_column = " ".join(target_column.split("_"))
    required_populations = required_populations if len(required_populations) > 0 else None
    dataset, class_id_to_counts, num_classes, class_to_id = init_dataset(hapt_genotypes_path=hapt_genotypes_path,
                                                                         extra_data_path=extra_data_path,
                                                                         target_column=target_column,
                                                                         minimum_samples=minimum_samples,
                                                                         with_extra_data=with_extra_data,
                                                                         required_populations=required_populations
                                                                         )
    # save class id map
    with open(os.path.join(experiment_results, 'class_id_map.json'), 'w') as f:
        json.dump(class_to_id, f)
    number_of_genotypes = dataset[0].shape[1]
    # save the reverse_dict for returning each id
    id_to_class = reverse_dict(class_to_id)
    print(f"dataset shapes: {dataset[0].shape, dataset[1].shape}")
    print(f"classes: {num_classes}, class_id_to_counts: {class_id_to_counts}")
    real_class_names = pd.DataFrame(np.argmax(dataset[1], 1))
    real_class_names = real_class_names[0].replace(id_to_class)
    real_class_names = 'Real_' + real_class_names
    real_class_names = list(real_class_names)

    #  build the Genome-AC-GAN including generator, discriminator and AC-GAN which is combine of both
    generator = build_generator(latent_dim=latent_size, num_classes=num_classes,
                                number_of_genotypes=number_of_genotypes, alph=alph)

    generator.compile(metrics=['accuracy'])
    discriminator = build_discriminator(number_of_genotypes=number_of_genotypes, num_classes=num_classes, alph=alph,
                                        d_activation=d_activation)

    if class_loss_function == "categorical_accuracy":
        class_loss_function = CategoricalAccuracy()
    if class_loss_function == "polyloss_ce":
        class_loss_function = polyloss_ce

    discriminator.compile(optimizer=RMSprop(lr=d_learn),
                          loss=[validation_loss_function, class_loss_function],
                          loss_weights=[1, class_loss_weights], metrics=['accuracy'])

    discriminator.trainable = False
    acgan = build_acgan(generator, discriminator)

    acgan.compile(optimizer=RMSprop(lr=g_learn),
                  loss=[validation_loss_function, class_loss_function],
                  loss_weights=[1, class_loss_weights],
                  metrics=['accuracy'])

    # prepare test_dataset for classification compression
    test_dataset = prepare_test_and_fake_dataset(experiment_results) if test_discriminator_classifier else None

    train(batch_size=batch_size, epochs=epochs, dataset=dataset, num_classes=num_classes,
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
    parser.add_argument('--target_column', type=str, default=DEFAULT_TARGET_COLUMN,
                        help='class column name', choices=['Population_code', 'Population_name',
                                                           'Superpopulation_code', 'Superpopulation_name'])
    parser.add_argument('--d_activation', type=str, default=DEFAULT_DISCRIMINATOR_ACTIVATION,
                        help='discriminator validation activation function (real/fake)')
    parser.add_argument('--class_loss_function', type=str, default=DEFAULT_CLASS_LOSS_FUNCTION,
                        help='loss function between different classes')
    parser.add_argument('--validation_loss_function', type=str, default=DEFAULT_VALIDATION_LOSS_FUNCTION,
                        help='loss function between different real/fake')
    parser.add_argument('--with_extra_data', action='store_true', default=False,
                        help="don't need to load extra data")
    parser.add_argument('--test_discriminator_classifier', action='store_true', default=False,
                        help="if you want to test the classifier during the training")
    parser.add_argument('--required_populations', nargs='+', help='List of specific populations to filter')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiment_results = os.path.join(DEFAULT_RESULTS_FOLDER, args.experiment_name)
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
         target_column=args.target_column,
         sequence_results_path=sequence_results,
         d_activation=args.d_activation,
         class_loss_function=args.class_loss_function,
         validation_loss_function=args.validation_loss_function,
         with_extra_data=args.with_extra_data,
         test_discriminator_classifier=args.test_discriminator_classifier,
         required_populations=args.required_populations)

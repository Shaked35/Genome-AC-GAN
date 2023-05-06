import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import tensorflow.python.keras.backend as K
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.layers import Dense, LeakyReLU
from tensorflow.python.keras.models import Sequential

from utils.util import init_dataset

plt.switch_backend('agg')
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.decomposition import PCA

output_folder = "experiment_results/old_model_80%"  # hapt format input file
hapt_genotypes_path = 'resource/train_0.8_super_pop.csv'
latent_size = 600  # size of noise input
alph = 0.01  # alpha value for LeakyReLU
g_learn = 0.0001  # generator learning rate
d_learn = 0.0008  # discriminator learning rate
epochs = 5001
batch_size = 256
save_that = 50  # epoch interval for saving outputs
target_column = "Superpopulation code"
Path(output_folder).mkdir(parents=True, exist_ok=True)


def init_gpus():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    enable_eager_execution()
    if gpus:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
            tensorflow.config.experimental.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tensorflow.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    else:
        print("No GPU device found")


init_gpus()

# Read input
(X_real, _), _, _, _ = init_dataset(hapt_genotypes_path=hapt_genotypes_path, target_column=target_column,
                                    without_extra_data=True)
df_pca = pd.DataFrame(X_real)
df_pca.columns = list(range(df_pca.shape[1]))
df_pca['Type'] = 'Real'
df_pca.to_csv("test_old_real.csv", index=False)
K.clear_session()

# Make generator
generator = Sequential()
generator.add(
    Dense(int(X_real.shape[1] // 1.2), input_shape=(latent_size,), kernel_regularizer=regularizers.l2(0.0001)))
generator.add(LeakyReLU(alpha=alph))
generator.add(Dense(int(X_real.shape[1] // 1.1), kernel_regularizer=regularizers.l2(0.0001)))
generator.add(LeakyReLU(alpha=alph))
generator.add(Dense(X_real.shape[1], activation='tanh'))

# Make discriminator
discriminator = Sequential()
discriminator.add(
    Dense(X_real.shape[1] // 2, input_shape=(X_real.shape[1],), kernel_regularizer=regularizers.l2(0.0001)))
discriminator.add(LeakyReLU(alpha=alph))
discriminator.add(Dense(X_real.shape[1] // 3, kernel_regularizer=regularizers.l2(0.0001)))
discriminator.add(LeakyReLU(alpha=alph))
discriminator.add(Dense(1, activation='sigmoid'))
# if gpu_count > 1:
#     discriminator = multi_gpu_model(discriminator, gpus=gpu_count)
discriminator.compile(optimizer=Adam(lr=d_learn), loss='binary_crossentropy')
# Set discriminator to non-trainable
discriminator.trainable = False

# Make GAN
gan = Sequential()
gan.add(generator)
gan.add(discriminator)

gan.compile(optimizer=Adam(lr=g_learn), loss='binary_crossentropy')

y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])

losses = []
# Training iteration
for e in range(1, epochs + 1):
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(X_real).shuffle(
        X_real.shape[0]).batch(batch_size, drop_remainder=True)
    for X_batch_real in train_dataset:
        X_batch_real = X_batch_real - np.random.uniform(0, 0.1, size=(X_batch_real.shape[0],
                                                                      X_batch_real.shape[1]))
        latent_samples = np.random.normal(loc=0, scale=1,
                                          size=(batch_size, latent_size))  # create noise to be fed to generator
        X_batch_fake = generator.predict_on_batch(latent_samples)  # create batch from generator using noise as input

        # train discriminator, notice that noise is added to real labels
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X_batch_real, y_real - np.random.uniform(0, 0.1, size=(
            y_real.shape[0], y_real.shape[1])))
        d_loss += discriminator.train_on_batch(X_batch_fake, y_fake)

        # make discriminator non-trainable and train gan
        discriminator.trainable = False
        g_loss = gan.train_on_batch(latent_samples, y_real)

    losses.append((d_loss, g_loss))
    print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (e + 1, epochs, d_loss, g_loss))
    if e % save_that == 0 or e == epochs:
        # Create AGs
        generated_genomes_total = []
        for i in range(int(1800 / batch_size)):
            latent_samples = np.random.normal(loc=0, scale=1, size=(batch_size, latent_size))
            generated_genomes = generator.predict(latent_samples)
            generated_genomes[generated_genomes < 0] = 0
            generated_genomes = np.rint(generated_genomes)
            tmp_generated_genomes_df = pd.DataFrame(generated_genomes)
            tmp_generated_genomes_df = tmp_generated_genomes_df.astype(int)
            generated_genomes_total.append(tmp_generated_genomes_df)
        generated_genomes_df = pd.concat(generated_genomes_total)
        generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
        generated_genomes_df['Type'] = "Fake"

        # Output AGs in hapt format
        generated_genomes_df.to_csv(os.path.join(output_folder, str(e) + "_output.hapt"), sep=" ",
                                    header=False, index=False)

        # Make PCA

        df_all_pca = pd.concat([df_pca, generated_genomes_df])
        df_all_pca = df_all_pca.reset_index(drop=True)  # reset index
        pca = PCA(n_components=2)
        PCs = pca.fit_transform(df_all_pca.drop('Type', axis=1))
        PCs_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
        PCs_df['Type'] = df_all_pca['Type']
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        pops = ['Real', 'Fake']
        colors = ['r', 'b']
        for pop, color in zip(pops, colors):
            ix = PCs_df['Type'] == pop
            ax.scatter(PCs_df.loc[ix, 'PC1']
                       , PCs_df.loc[ix, 'PC2']
                       , c=color
                       , s=50, alpha=0.2)
        ax.legend(pops)
        fig.savefig(os.path.join(output_folder, str(e) + '_pca.pdf'), format='pdf')

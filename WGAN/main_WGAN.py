import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

from models_10K import *  # models_10K for 16383 zero padded SNP data
from pca_plot_genomes import pca_plot
from utils.consts import *
from utils.util import init_dataset


class CustomDataset(Dataset):
    def __init__(self, X_values, Y_values):
        self.X_values = X_values
        self.Y_values = Y_values
        self.length = len(X_values)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X_values[idx]
        y = self.Y_values[idx]
        return x, y


def main(hapt_genotypes_path: str, extra_data_path: str, experiment_results_path: str, latent_size: int, alph: float,
         g_learn: float, d_learn: float, epochs: int, batch_size: int, class_loss_weights: float, save_number: int,
         minimum_samples: int, target_column: str, sequence_results_path: str, d_activation: str,
         class_loss_function: str, validation_loss_function: str, with_extra_data: bool,
         test_discriminator_classifier: bool, required_populations: list[str]):
    ## Set seed for reproducibility
    target_column = " ".join(target_column.split("_"))
    required_populations = required_populations if required_populations is not None and len(
        required_populations) > 0 else None
    dataset, class_id_to_counts, num_classes, class_to_id = init_dataset(
        hapt_genotypes_path=f"../{hapt_genotypes_path}",
        extra_data_path=extra_data_path,
        target_column=target_column,
        minimum_samples=minimum_samples,
        with_extra_data=with_extra_data,
        required_populations=required_populations
    )
    manualSeed = 9
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed: ", manualSeed)

    # inpt = "../1000G_real_genomes/10K_SNP_1000G_real.hapt"  # hapt format input file
    # out_dir = "./output_dir"
    # alph = 0.01  # alpha value for LeakyReLU
    # g_learn = 0.0005  # generator learning rate
    # d_learn = 0.0005  # discriminator learning rate
    # epochs = 10001
    # batch_size = 16
    channels = 10  # channel multiplier which dictates the number of channels for all layers
    ag_size = 500  # number of artificial genomes (haplotypes) to be created
    gpu = 0  # number of GPUs
    # save_number = 50  # epoch interval for saving outputs
    pack_m = 3  # packing amount for the critic
    critic_iter = 10  # number of times critic is trained for every generator training
    label_noise = 1  # noise for real labels (1: noise, 0: no noise)
    noise_dim = 2  # dimension of noise for each noise vector
    latent_depth_factor = 12  # 14 for 65535 SNP data and 12 for 16383 zero padded SNP data

    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")  # set device to cpu or gpu

    ## Prepare the training data
    df = pd.read_csv(f"../{hapt_genotypes_path}", header=None)
    df = df.sample(frac=1).reset_index(drop=True)
    # df_noname = df.drop(df.columns[0:2], axis=1)
    # df_noname = df_noname.values
    # df_noname = df_noname - np.random.uniform(0,0.1, size=(df_noname.shape[0], df_noname.shape[1]))
    # df = df.iloc[0:ag_size, :]
    df_noname = torch.Tensor(dataset[0])
    df_noname = df_noname.to(device)

    dataset = CustomDataset(dataset[0], dataset[1])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    latent_size = int((df_noname.shape[1] + 1) / (2 ** latent_depth_factor))  # set the latent_size
    # latent_size = df_noname.shape[1]
    ## Create the generator
    netG = ConvGenerator(latent_size=latent_size, data_shape=df_noname.shape[1], gpu=gpu, device=device,
                         channels=channels, noise_dim=noise_dim, alph=alph, num_classes=num_classes)
    netG = netG.float()
    if (device.type == 'cuda') and (gpu > 1):
        netG = nn.DataParallel(netG, list(range(gpu)))
    netG.to(device)

    ## Create the critic
    netC = ConvDiscriminator(data_shape=df_noname.shape[1], latent_size=latent_size, gpu=gpu, pack_m=pack_m,
                             device=device, channels=channels, alph=alph).to(device)
    netC = netC.float()
    if (device.type == 'cuda') and (gpu > 1):
        netC = nn.DataParallel(netC, list(range(gpu)))
    netC.to(device)

    ## Optimizers for generator and critic
    c_optimizer = torch.optim.Adam(netC.parameters(), lr=d_learn, betas=(0.5, 0.9))
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=g_learn, betas=(0.5, 0.9))

    label_fake = torch.tensor(1, dtype=torch.float).to(device)
    label_real = label_fake * -1
    losses = []

    ## Noise generator function to be used to provide input noise vectors
    def noise_generator(size, noise_count, noise_dim, device, class_size=30):
        noise_list = []
        for i in range(2, noise_count * 2 + 1, 2):
            noise = torch.normal(mean=0, std=1, size=(size, noise_dim, (latent_size + class_size) * (2 ** i) - 1),
                                 device=device)
            noise_list.append(noise)
        return noise_list

    ## Training Loop
    print("Starting Training Loop...")
    start_time = time.time()
    for epoch in range(epochs):
        # c_loss_real = 0
        # c_loss_fake = 0
        print(f"epoch number {epoch}")

        b = 0

        while b < len(dataloader):
            for param in netC.parameters():
                param.requires_grad = True

            # Update Critic
            for n_critic in range(critic_iter):

                netC.zero_grad(set_to_none=True)

                X_batch_real, Y_batch_real = next(iter(dataloader))
                X_batch_real = torch.reshape(X_batch_real, (X_batch_real.shape[0], 1, X_batch_real.shape[1]))

                if pack_m > 1:
                    for i in range(pack_m - 1):
                        temp_x_batch, temp_y_batch = next(iter(dataloader))
                        temp_x_batch = torch.reshape(temp_x_batch, (temp_x_batch.shape[0], 1, temp_x_batch.shape[1]))
                        X_batch_real = torch.cat((X_batch_real, temp_x_batch), 1)
                b += 1

                # Train Critic with real samples
                c_loss_real = netC(X_batch_real.float())
                c_loss_real = c_loss_real.mean()

                if label_noise != 0:
                    label_noise = torch.tensor(random.uniform(0, 0.1), dtype=torch.float, device=device)
                    c_loss_real.backward(label_real + label_noise)
                else:
                    c_loss_real.backward(label_real)

                for i in range(pack_m):
                    latent_samples = torch.normal(mean=0, std=1, size=(batch_size, noise_dim, latent_size),
                                                  device=device)  # create the initial noise to be fed to generator
                    noise_list = noise_generator(batch_size, 6, noise_dim, device)
                    fake_labels_batch = torch.nn.functional.one_hot(
                        torch.randint(0, num_classes, (batch_size,)), 30)
                    temp = netG(latent_samples, fake_labels_batch, noise_list)
                    if i == 0:
                        X_batch_fake = temp
                    else:
                        X_batch_fake = torch.cat((X_batch_fake, temp), 1)

                c_loss_fake = netC(X_batch_fake.detach())
                c_loss_fake = c_loss_fake.mean()
                c_loss_fake.backward(label_fake)

                # Train with gradient penalty
                gp = gradient_penalty(netC, X_batch_real.float(), X_batch_fake.float(), device)
                gp.backward()
                c_loss = c_loss_fake - c_loss_real + gp
                c_optimizer.step()

            for param in netC.parameters():
                param.requires_grad = False

            # Update G network
            netG.zero_grad(set_to_none=True)
            for i in range(pack_m):
                latent_samples = torch.normal(mean=0, std=1, size=(batch_size, noise_dim, latent_size),
                                              device=device)  # create the initial noise to be fed to generator
                noise_list = noise_generator(batch_size, 6, noise_dim, device)
                fake_labels_batch = torch.nn.functional.one_hot(
                    torch.randint(0, num_classes, (batch_size,)), 30)
                temp = netG(latent_samples, fake_labels_batch, noise_list)
                if i == 0:
                    X_batch_fake = temp
                else:
                    X_batch_fake = torch.cat((X_batch_fake, temp), 1)

            g_loss = netC(X_batch_fake)
            g_loss = g_loss.mean()
            g_loss.backward(label_real)
            g_optimizer.step()

            # Save Losses for plotting later
            # losses.append((c_loss.item(), g_loss.item()))
            losses.append((round(c_loss.item(), 3), (round(g_loss.item(), 3))))

        ## Outputs for assessment at every "save_number" epoch
        if epoch % save_number == 0 or epoch == epochs:
            torch.cuda.empty_cache()
            torch.save({
                'Generator': netG.state_dict(),
                'Critic': netC.state_dict(),
                'G_optimizer': g_optimizer.state_dict(),
                'C_optimizer': c_optimizer.state_dict()},
                f'{sequence_results_path}/{epoch}')

            netG.eval()
            latent_samples = torch.normal(mean=0, std=1, size=(ag_size, noise_dim, latent_size),
                                          device=device)  # create the initial noise to be fed to generator
            noise_list = noise_generator(ag_size, 6, noise_dim, device)
            with torch.no_grad():
                fake_labels_batch = torch.nn.functional.one_hot(
                    torch.randint(0, num_classes, (ag_size,)), 30)
                generated_genomes = netG(latent_samples, fake_labels_batch, noise_list)
            generated_genomes = generated_genomes.cpu().detach().numpy()
            generated_genomes[generated_genomes < 0] = 0
            generated_genomes = np.rint(generated_genomes)
            generated_genomes_df = pd.DataFrame(np.reshape(generated_genomes, (ag_size, generated_genomes.shape[2])))
            generated_genomes_df = generated_genomes_df.astype(int)
            gen_names = list()

            for i in range(0, len(generated_genomes_df)):
                gen_names.append('AG' + str(i))
            generated_genomes_df.insert(loc=0, column='Type', value="AG")
            generated_genomes_df.insert(loc=1, column='ID', value=gen_names)
            generated_genomes_df.columns = list(range(generated_genomes_df.shape[1]))
            df.columns = list(range(df.shape[1]))

            # Output AGs in hapt or hdf formats
            # generated_genomes_df.to_csv(f'{out_dir}/{epoch}_output.hapt', sep=" ", header=False, index=False)
            generated_genomes_df.to_hdf(f'{sequence_results_path}/{epoch}_output.hapt', key="df1", mode="w")

            # Output lossess
            pd.DataFrame(losses).to_csv(f'{sequence_results_path}/{epoch}_losses.txt', sep=" ", header=False,
                                        index=False)
            fig, ax = plt.subplots()
            plt.plot(np.array([losses]).T[0], label='Critic')
            plt.plot(np.array([losses]).T[1], label='Generator')
            plt.title("Training Losses")
            plt.legend()
            fig.savefig(f'{sequence_results_path}/{epoch}_loss.pdf', format='pdf')

            # Plot PCA
            pca_plot(df, generated_genomes_df, epoch, dir=sequence_results_path)

            netG.train()
    print("--- %s seconds ---" % (time.time() - start_time))


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

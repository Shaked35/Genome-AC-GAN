REAL_10K_SNP_1000G_PATH = "resource/10K_SNP_1000G_real.hapt"
REAL_EXTRA_DATA_PATH = "resource/extra_data_1000G.tsv"
DEFAULT_RESULTS_FOLDER = "experiment_results"
SAMPLE_COLUMN_NAME = "Sample name"
DEFAULT_LATENT_SIZE = 600  # size of noise input
DEFAULT_ALPH = 0.01  # alpha value for LeakyReLU
DEFAULT_GENERATOR_LEARNING_RATE = 0.0001  # generator learning rate
DEFAULT_DISCRIMINATOR_LEARNING_RATE = 0.0008  # discriminator learning rate
REGULARIZERS_FACTOR = 0.0001
GENERATOR_POP_FACTOR = 1
EPOCH_LR_SCHEDULE = 5000
DEFAULT_EPOCHS = 10001
DEFAULT_BATCH_SIZE = 256
DEFAULT_SAVE_NUMBER = 500  # epoch interval for saving outputs
DEFAULT_CLASS_LOSS_WEIGHTS = 1
DEFAULT_MINIMUM_SAMPLES = 50
DEFAULT_FIRST_EPOCH = 1
DEFAULT_TARGET_COLUMN = "Superpopulation_code"
DEFAULT_EXPERIMENT_NAME = "default_genome-ac-gan"
SEQUENCE_RESULTS_PATH = "fake_genotypes_sequences/new_sequences"
DEFAULT_EXPERIMENT_OUTPUT_DIR = "comparison_model_tests"
REAL_POSITION_FILE_NAME = "resource/10k_SNP.legend"
MODEL_CONFIG_PATH = "model_name_to_path.json"
DEFAULT_DISCRIMINATOR_ACTIVATION = "sigmoid"
DEFAULT_CLASS_LOSS_FUNCTION = "polyloss_ce"
DEFAULT_VALIDATION_LOSS_FUNCTION = "binary_cros sentropy"
NUMBER_REPEAT_CLASS_VECTOR = 8
DISCRIMINATOR_METRICS_FILE = "categorical_crossentropy_02.csv"
CONFUSION_MATRIX_FILE = "confusion_matrix.jpg"
DEFAULT_EPSILON_LCE = 0.2
DEFAULT_ALPH_LCE = 0.1

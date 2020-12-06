TRAIN_DATA_PATH = "./data/train"
VAL_DATA_PATH = "./data/val"
TEST_DATA_PATH = "./data/test"
LABELS = ['BACTERIA', 'VIRUS', 'NORMAL']
# LABELS = ["PNEUMONIA", "NORMAL"]  # Binary Classification
IMG_SIZE = 150
GRAYSCALE = False

# Model training hyperparameters
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
OPTIMIZER = "Adam"
LR_REDUCTION = True
EARLY_STOPPING = False

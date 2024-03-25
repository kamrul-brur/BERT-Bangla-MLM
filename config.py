from dataclasses import dataclass
@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 10000
    EMBED_DIM = 128
    NUM_HEAD = 8
    FF_DIM = 128
    NUM_LAYERS = 1
    EPOCHS = 100
    DATASET_PATH = "./dataset/dataset.txt"
    LOG_DIRECTORY = "logs"
    SAVED_MODEL_NAME = "bert_mlm_bangla.keras"
    SAVED_VECTORIZED_LAYER_NAME = "vectorizer_layer.pkl"
    TENSORBOARD_LOG_DIR = "logs/fit"
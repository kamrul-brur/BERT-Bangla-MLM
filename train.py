from config import *
from bert_module import *
import utils
import data_preprocess
import argparse
import re
import os
import glob
import string
import pickle
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization

# get the config parameters
config = Config()


def train(args):
    """
    Train the BERT model for masked language modeling.

    This function performs the following steps:
    1. Loads the dataset from the specified text files.
    2. Preprocesses the data by tokenizing and masking the tokens.
    3. Creates a vectorize layer to convert tokens to embeddings.
    4. Prepares the masked language modeling dataset.
    5. Defines callbacks for generating sample text and logging to TensorBoard.
    6. Creates and trains the BERT model.
    7. Saves the trained model and vectorized layer.

    Args:
        args: An argparse namespace containing the following arguments:
            - text_path (str): Path to the directory containing the text files.
            - model_log_dir (str): Directory for saving model logs.
            - model_name (str): Name of the saved model.
            - vectorized_layer_name (str): Name of the vectorized layer.
            - sample_text_for_callback (str): Sample text used in the callback.

    Returns:
        None
    """
    text_path =args.text_path
    model_log_dir=args.model_log_dir
    model_name =args.model_name
    vectorized_layer_name =args.vectorized_layer_name
    sample_text_for_callback = args.sample_text_for_callback
    all_data = data_preprocess.get_data_from_text_files(text_path)
    vectorize_layer = utils.get_vectorize_layer(
        all_data.text.values.tolist(),
        config.VOCAB_SIZE,
        config.MAX_LEN,
        special_tokens=["<blank>"],
    )
    mask_token_id = vectorize_layer(["<blank>"]).numpy()[0][0]
    x_all_text = utils.encode(all_data.text.values, vectorize_layer)
    x_masked_train, y_masked_labels, sample_weights = utils.get_masked_input_and_labels(
        x_all_text, mask_token_id
    )
    mlm_ds = tf.data.Dataset.from_tensor_slices(
        (x_masked_train, y_masked_labels, sample_weights)
    )
    mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
    id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
    token2id = {y: x for x, y in id2token.items()}
    sample_tokens = vectorize_layer([sample_text_for_callback])
    generator_callback = MaskedTextGenerator(sample_tokens, mask_token_id, id2token)
    if Path(config.TENSORBOARD_LOG_DIR).exists() and Path(config.TENSORBOARD_LOG_DIR).is_dir():
        shutil.rmtree(Path(config.TENSORBOARD_LOG_DIR))
    log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    bert_masked_model = create_masked_language_bert_model()
    bert_masked_model.summary()
    print('Training Start.....')
    bert_masked_model.fit(mlm_ds, epochs=config.EPOCHS, callbacks=[generator_callback, tensorboard_callback])
    pickle.dump({'config': vectorize_layer.get_config(),
                 'weights': vectorize_layer.get_weights()}, open(os.path.join(model_log_dir, vectorized_layer_name), "wb"))
    bert_masked_model.save(os.path.join(model_log_dir, model_name))
    print(f"Model successfully saved into {model_log_dir}/.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model with necessary arguments.")
    parser.add_argument('--text_path', type=str, default=config.DATASET_PATH,
                        help='Path to the text file')
    parser.add_argument('--model_name', type=str, default=config.SAVED_MODEL_NAME,
                        help='Name of the saved model')
    parser.add_argument('--vectorized_layer_name', type=str, default=config.SAVED_VECTORIZED_LAYER_NAME,
                        help='Name of the vectorized layer')
    parser.add_argument('--model_log_dir', type=str, default=config.LOG_DIRECTORY,
                        help='Directory for saving model logs')
    parser.add_argument('--sample_text_for_callback', type=str, default="আমার সোনার বাংলা <blank> তোমায় ভালবাসি ",
                        help='Sample text used in callback')
    args = parser.parse_args()
    os.makedirs(args.model_log_dir, exist_ok=True)
    train(args)

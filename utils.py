import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization


def custom_standardization(input_data):
    """pre processing for text vectorization. Removes html tags, new lines, and other punctuation

    Args:
        input_data (str): text input

    Returns:
        str: clean text
    """
   # lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    stripped_new_line = tf.strings.regex_replace(stripped_html, "\n", " ")
    return tf.strings.regex_replace(
        stripped_new_line, "[%s]" % re.escape("!#$%&'()*+,-./:;=?@\^_`{|}~"), ""
    )


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["<blank>"]
    vectorize_layer.set_vocabulary(vocab)
    vocab = vectorize_layer.get_vocabulary()
    return vectorize_layer


def encode(texts, vectorize_layer):
    """vectorize input text for encoding

    Args:
        texts (str): input text
        vectorize_layer (TextVectorization Layer): keras layer

    Returns:
        numpy array: encoded text
    """
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()


def get_masked_input_and_labels(encoded_texts, mask_token_id):
    """mask input data (15%), 

    Args:
        encoded_texts (numpy array): encoded text from text vectorizer
        mask_token_id (int): token id of <blank> 

    Returns:
        encoded_texts_masked, y_labels, sample_weights
    """
 
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]
    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to <blank> which is the last token for the 90% of tokens
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0
    y_labels = np.copy(encoded_texts)
    return encoded_texts_masked, y_labels, sample_weights


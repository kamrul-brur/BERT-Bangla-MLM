from config import * 

import re
import glob
import string
import numpy as np
import pandas as pd
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass


config = Config()

def bert_module(query, key, value, i):
    """
    Defines the BERT module, which consists of a multi-headed self-attention layer followed by a feed-forward layer.

    Args:
        query (tensor): The query tensor for the multi-headed attention layer.
        key (tensor): The key tensor for the multi-headed attention layer.
        value (tensor): The value tensor for the multi-headed attention layer.
        i (int): The index of the BERT module.

    Returns:
        sequence_output (tensor): The output tensor of the BERT module, which is the sum of the attention output and the feed-forward output, normalized using layer normalization.
    """
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    """
    Generate the positional encoding matrix.

    Args:
        max_len (int): Maximum length of the sequence.
        d_emb (int): Dimension of the embedding.

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (max_len, d_emb).
    """
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")


class MaskedLanguageModel(tf.keras.Model):
    """
    A custom Keras model for masked language modeling.

    This model defines the training step, which includes the following:
    - Forward pass through the model
    - Computation of the loss
    - Computation of gradients
    - Updating of weights using an optimizer
    - Tracking the loss metric
    """
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker]


def create_masked_language_bert_model():
    """Define the Bert model with layers, compile the model, and return it.

    Returns:
        Keras Model: The created masked language Bert model.
    """
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model



class MaskedTextGenerator(keras.callbacks.Callback):
    """This is used for callback while training. At the end of an epoch, it shows the prediction result on a sample text

    Args:
        sample_tokens (Tensor): The sample tokens used for prediction.
        mask_token_id (int): The mask token id.
        id2token (dict): The token to id mapping dictionary.
        top_k (int): The number of top predictions to show. Default is 5.
    """
    def __init__(self, sample_tokens, mask_token_id, id2token, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k
        self.mask_token_id = mask_token_id
        self.id2token = id2token

    def decode(self, tokens):
        return " ".join([self.id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return self.id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens.numpy())

        masked_index = np.where(self.sample_tokens.numpy() == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy( self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode( self.sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)

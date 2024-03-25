import bert_module
import config

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pprint import   pprint
from keras.layers import TextVectorization

config = config.Config()

def load_models(bert_masked_model_path, vectorize_layer_path):
    """load the trained saved model and vectorized layer

    Args:
        bert_masked_model_path (str): bert model path
        vectorize_layer_path (str): vectorized layer pickle file path
    """

    # load vectorized layer
    from_disk = pickle.load(open(vectorize_layer_path, "rb"))
    vectorize_layer = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorize_layer.set_weights(from_disk['weights'])
    # Load pretrained bert model
    bert_masked_model = keras.models.load_model(
    bert_masked_model_path, custom_objects={"MaskedLanguageModel": bert_module.MaskedLanguageModel}
    )
    

    return vectorize_layer, bert_masked_model



def predict(input_text, bert_masked_model, vectorize_layer):
    """predict masked text

    Args:
        input_text (str): input text
        bert_masked_model (keras model): trained model
        vectorize_layer (textvectorizer layer): vectorizer layerthat contains vocab
    """
    # id to token convert
    id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
    #token 2 id convert
    token2id = {y: x for x, y in id2token.items()}
    # tokenize input text
    sample_tokens = vectorize_layer([input_text])
    #sample_tokens = sample_tokens.numpy()
    k = 5
    # calling predict function
    prediction = bert_masked_model.predict(sample_tokens)
    # get token id of <blank>
    mask_token_id = vectorize_layer(["<blank>"]).numpy()[0][0]
    masked_index = np.where(sample_tokens == mask_token_id)
    masked_index = masked_index[1]
    mask_prediction = prediction[0][masked_index]
    # top predictions
    top_indices = mask_prediction[0].argsort()[-k :][::-1]
    values = mask_prediction[0][top_indices]
   # get the prediction with highest probability
    p = top_indices[0]
    v = values[0]
    tokens = np.copy(sample_tokens[0])
    tokens[masked_index[0]] = p
    # show prediction
    print("Input text: ", input_text)
    print("Predicted Text: ", decode(tokens,id2token))
    print("\n"*2)





def decode( tokens, id2token):
    """return string from tokens

    Args:
        tokens (numpy array): encoded tokens
        id2token : id to token mapper

    Returns:
        str: string from tokens
    """
    return " ".join([id2token[t] for t in tokens if t != 0])

def convert_ids_to_tokens( id, id2token):
    """return tokens from ids

    Args:
        id : id of the word from vocab
        id2token : id to token mapper

    Returns:
        _type_: _description_
    """
    return id2token[id]





if __name__ == '__main__':
    # change input text path
    input_text_path = "dataset/inputs.txt"
    #input_text = "এখান থেকেও পরিষ্কার <blank> যাচ্ছে"
    #path of the saved vectorized layer
    vectorized_layer_path = os.path.join(config.LOG_DIRECTORY, config.SAVED_VECTORIZED_LAYER_NAME)
    #path of the saved model
    model_path = os.path.join(config.LOG_DIRECTORY, config.SAVED_MODEL_NAME)
    #load the model and the vectorizer
    vectorize_layer, bert_masked_model = load_models(model_path, vectorized_layer_path)
    # open file and predict text
    with open(input_text_path, encoding="utf8") as input:
        for line in input:
            if line.find("<blank>") != -1:
                predict(line, bert_masked_model, vectorize_layer)


    



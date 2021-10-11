import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #    [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
    # 
    # - When computing loss, the decoder labels should have the first word removed:
    #    [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 
    
    train_english_call = train_english[:, 0:train_english.shape[1]-1]
    train_english_loss = train_english[:, 1:train_english.shape[1]]
    
    num_examples = train_french.shape[0]
    shuffle_indices = np.arange(0, num_examples)
    shuffle_indices = tf.random.shuffle(shuffle_indices)
    train_french = tf.gather(train_french, shuffle_indices)
    train_english_call = tf.gather(train_english_call, shuffle_indices)
    train_english_loss = tf.gather(train_english_loss, shuffle_indices)

    optimizer = tf.keras.optimizers.Adam(model.learning_rate)

    for i in range(0, num_examples, model.batch_size):
        french_batch = train_french[i:i + model.batch_size, :]
        english_call_batch = train_english_call[i:i + model.batch_size, :]
        english_loss_batch = train_english_loss[i:i + model.batch_size, :]
        
        with tf.GradientTape() as tape:
            probs = model.call(french_batch, english_call_batch)
            mask = np.where(english_loss_batch == eng_padding_index, 0, 1)
            loss = model.loss_function(probs, english_loss_batch, mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@av.test_func
def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
    e.g. (my_perplexity, my_accuracy)
    """

    # Note: Follow the same procedure as in train() to construct batches of data!
    test_english_call = test_english[:, 0:test_english.shape[1]-1]
    test_english_loss = test_english[:, 1:test_english.shape[1]]

    perplexities = 0.0
    accuracies = 0.0
    symbols = 0.0
    for i in range(0, test_french.shape[0], model.batch_size):
        french_batch = test_french[i:i + model.batch_size, :]
        english_call_batch = test_english_call[i:i + model.batch_size, :]
        english_loss_batch = test_english_loss[i:i + model.batch_size, :]
        
        probs = model.call(french_batch, english_call_batch)
        mask = np.where(english_loss_batch == eng_padding_index, 0, 1)
        batch_symbols = np.sum(mask)
        perplexities += model.loss_function(probs, english_loss_batch, mask)
        accuracies += (model.accuracy_function(probs, english_loss_batch, mask) * batch_symbols)
        symbols += batch_symbols

    return (np.exp(perplexities/symbols), accuracies/symbols)

def main():	
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
            print("USAGE: python assignment.py <Model Type>")
            print("<Model Type>: [RNN/TRANSFORMER]")
            exit()

    # Change this to "True" to turn on the attention matrix visualization.
    # You should turn this on once you feel your code is working.
    # Note that it is designed to work with transformers that have single attention heads.
    if sys.argv[1] == "TRANSFORMER":
        av.setup_visualization(enable=False)

    print("Running preprocessing...")
    train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../../data/fls.txt','../../data/els.txt','../../data/flt.txt','../../data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args) 

    # TODO:
    # Train and Test Model for 1 epoch.
    train(model, train_french, train_english, eng_padding_index)
    perplexity, accuracy = test(model, test_french, test_english, eng_padding_index)
    print('perplexity: ' + str(perplexity))
    print('accuracy: ' + str(accuracy))

    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    av.show_atten_heatmap()

if __name__ == '__main__':
    main()
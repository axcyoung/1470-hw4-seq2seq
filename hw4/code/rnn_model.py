import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        ###### DO NOT CHANGE ##############
        super(RNN_Seq2Seq, self).__init__()
        self.french_vocab_size = french_vocab_size # The size of the french vocab
        self.english_vocab_size = english_vocab_size # The size of the english vocab

        self.french_window_size = french_window_size # The french window size
        self.english_window_size = english_window_size # The english window size
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters
        self.learning_rate = .01

        # Define batch size and optimizer/learning rate
        self.batch_size = 100 # You can change this
        self.embedding_size = 128 # You should change this

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.english_E = tf.Variable(tf.random.truncated_normal([self.english_vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.french_E = tf.Variable(tf.random.truncated_normal([self.french_vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.encoder = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.english_vocab_size)


    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # TODO:
        #1) Pass your french sentence embeddings to your encoder 
        #2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
        #3) Apply dense layer(s) to the decoder out to generate probabilities

        french_embedding = tf.nn.embedding_lookup(self.french_E, encoder_input)
        encoder_fstate = self.encoder(french_embedding)
        english_embedding = tf.nn.embedding_lookup(self.english_E, decoder_input)
        decoder_fseq = self.decoder(english_embedding, initial_state=(encoder_fstate[1], encoder_fstate[2]))
        logits = self.dense(decoder_fseq[0])
        probs = tf.nn.softmax(logits)
        return probs

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the total model cross-entropy loss after one forward pass. 
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
        losses = tf.boolean_mask(losses, mask)
        return tf.reduce_sum(losses)
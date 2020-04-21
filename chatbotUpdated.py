import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
import unicodedata
import re
import io
import time


tf.compat.v1.enable_eager_execution()

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Loading the conversations file from the dataset
lines = open("./cornell movie-dialogs corpus/movie_lines.txt", encoding = "utf-8", errors = "ignore").read().split("\n")
conversations = open("./cornell movie-dialogs corpus/movie_conversations.txt", encoding = "utf-8", errors = "ignore").read().split("\n")

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if (len(_line) == 5):
        id2line[_line[0]] = _line[4]

conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(","))

questions = []
answers = []

for conversation_id in conversation_ids:
    for i in range(len(conversation_id) - 1):
        questions.append(id2line[conversation_id[i]])
        answers.append(id2line[conversation_id[i + 1]])

def clean_text(text):
    text = text.lower()
    text = unicode_to_ascii(text.strip())
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    text = '<start> ' + text + ' <end>'
    return text

def tokenize(text):
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
    text_tokenizer.fit_on_texts(text)
    tensor = text_tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding = 'post')
    return tensor, text_tokenizer

clean_questions = []
clean_answers = []

def load_dataset():
    for question in questions:
        clean_questions.append(clean_text(question))
    for answer in answers:
        clean_answers.append(clean_text(answer))
    input_tensor, input_text_tokenizer = tokenize(clean_questions)
    target_tensor, target_text_tokenizer = tokenize(clean_answers)
    return input_tensor, input_text_tokenizer, target_tensor, target_text_tokenizer

input_tensor, input_lang, target_tensor, target_lang = load_dataset()

# Calculating the max length of the target tensor...
max_len_target, max_len_input = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size = 0.2)



# Create a tf.data dataset
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 100
steps_per_epoch = len(input_tensor_train)
embedding_dim = 256
units = 1024
vocab_inp_size = len(input_lang.word_index) + 1
vocab_targ_size = len(target_lang.word_index) + 1


dataset= tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis = 1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis = 1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding =  tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences= True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)


    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(vocab_targ_size, embedding_dim, units, BATCH_SIZE)

# Defining the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()
loss_object= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype = loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoder = encoder, decoder = decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:,t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss/ int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if (batch % 100 == 0):
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

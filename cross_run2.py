import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

tf.config.run_functions_eagerly(True)
def set_seeds(seed=1299827):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

#### user encoding ####
names = ['occupations']
data_dir = "./data/ml-100k"
occupations = pd.read_csv(os.path.join(data_dir, 'u.occupation'), '\t', names=names,
                       engine='python')

names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
data_dir = "./data/ml-100k"
user_info = pd.read_csv(os.path.join(data_dir, 'u.user'), '|', names=names,
                       engine='python')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
    return ds

def get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

user_info.drop(columns=['user_id'], inplace=True)
gender = {'M': 0, 'F': 1}
user_info['gender'] = user_info['gender'].apply(lambda x: gender[x])

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)

binary_feature_names = ['gender']
categorical_feature_names = ['occupation', 'zip_code']
numeric_features = user_info[['age']]

inputs = {}
for name, column in user_info.items():
#     if type(column[0]) == str:
#         dtype = tf.string
    if (name in categorical_feature_names or
        name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)


preprocessed = []

for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

users = tf.convert_to_tensor(user_info['gender'].values, dtype=tf.float32)
users = tf.concat((users[:, tf.newaxis], tf.squeeze(normalizer(user_info['age'].values))[:, tf.newaxis]), axis=1)

numeric_inputs = {}
for name in ['age']:
    numeric_inputs[name]=inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

for name in categorical_feature_names:
    vocab = sorted(set(user_info[name]))
    print(f'name: {name}')
#     print(f'vocab: {vocab}\n')

    if type(vocab[0]) is str:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot', num_oov_indices=0)
    else:
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot', num_oov_indices=0)

    x = lookup(user_info[name].values)
    users = tf.concat((users, x), axis=1)

encoding_dim = 32
user_input = keras.Input(shape=(818,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(128, activation='leaky_relu')(user_input)
encoded = layers.Dense(64, activation='leaky_relu')(encoded)
encoded = layers.Dense(32, activation='leaky_relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(64, activation='leaky_relu')(encoded)
decoded = layers.Dense(128, activation='leaky_relu')(decoded)
decoded = layers.Dense(818)(decoded)

user_autoencoder = keras.Model(user_input, decoded)
user_autoencoder.compile(optimizer='adam', loss='mse')

user_encoder = keras.Model(user_input, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_output = user_autoencoder.layers[-3](encoded_input)
decoder_output = user_autoencoder.layers[-2](decoder_output)
decoder_output = user_autoencoder.layers[-1](decoder_output)

# Create the decoder model
user_decoder = keras.Model(encoded_input, decoder_output)

user_autoencoder.compile(optimizer='adam', loss='mse')


### encode item ###
from transformers import AutoTokenizer
from transformers import TFAutoModel
from keras.optimizers import Adam

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
names = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
              'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]data_dir = "./data/ml-100k"
item_info = pd.read_csv(os.path.join(data_dir, 'u.item'), '|', names=names,
                       engine='python', encoding = "ISO-8859-1", index_col=False)

item_info.drop(columns=['movie_id', 'video_release_date', 'IMDb_URL'], inplace=True)
item_info['movie_title'] = item_info['movie_title'].str.replace(r" \(\d{4}\)","")

lm_input = tokenizer(item_info['movie_title'].values.tolist(), return_tensors='tf', truncation=True, padding=True, max_length=8)

model = TFAutoModel.from_pretrained("roberta-base")
lm_output = model.predict(lm_input)

items = tf.reshape(lm_output[0], (1682, -1))
item_info['release_year'] = item_info['release_date'].str.extract(r'(\d{4}$)')
item_info['release_year'] = item_info['release_year'].apply(lambda x: int(x) if pd.notnull(x) else x)
item_info['day_sin'] = np.sin(2 * np.pi * (pd.to_datetime(item_info['release_date']).dt.dayofyear/365.0))
item_info['day_cos'] = np.cos(2 * np.pi * (pd.to_datetime(item_info['release_date']).dt.dayofyear/365.0))
item_info.drop(columns=['release_date'], inplace=True)
item_info.fillna(item_info.mean(), inplace=True)
items = tf.concat((items, tf.convert_to_tensor(item_info.iloc[:, 1:20], dtype=tf.float32)),axis=1)
numeric_features = item_info.columns.tolist()

inputs = {}
for name, column in item_info.items():
#     if type(column[0]) == str:
#         dtype = tf.string
    if (name in categorical_feature_names or
        name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

preprocessed = []

for name in numeric_features[:-3]:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)

numeric_inputs = {}
for name in ['release_year']:
    numeric_inputs[name]=inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

items = tf.concat((items, tf.squeeze(normalizer(item_info['release_year'].values))[:,tf.newaxis]), axis=1)
items = tf.concat((items, tf.convert_to_tensor(item_info.iloc[:,-2:], dtype=tf.float32)), axis=1)

for name in numeric_features[-2:]:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)

encoding_dim = 32
item_input = keras.Input(shape=(6166,))
# "encoded" is the encoded representation of the input
encoded = tf.keras.layers.BatchNormalization()(item_input)
encoded = layers.Dense(1024, activation='leaky_relu')(encoded)
encoded = layers.Dense(128, activation='leaky_relu')(encoded)
encoded = layers.Dense(32, activation='leaky_relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(128, activation='leaky_relu')(encoded)
decoded = layers.Dense(1024, activation='leaky_relu')(decoded)
decoded = layers.Dense(6166)(decoded)


item_autoencoder = keras.Model(item_input, decoded)
item_autoencoder.compile(optimizer='adam', loss='mse')

item_encoder = keras.Model(item_input, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_output = item_autoencoder.layers[-3](encoded_input)
decoder_output = item_autoencoder.layers[-2](decoder_output)
decoder_output = item_autoencoder.layers[-1](decoder_output)

# Create the decoder model
item_decoder = keras.Model(encoded_input, decoder_output)

item_autoencoder.compile(optimizer='adam', loss='mse')

### embedding extraction ###

user_ids = []
item_ids = []
ratings = []
timestamps = []

with open("./data/ml-100k/u.data", 'rt') as file1:
    for line in file1.readlines():
        a = line.split()
        user_ids.append(a[0])
        item_ids.append(a[1])
        ratings.append(a[2])
        timestamps.append(a[3])
    
rating_df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings, 'timestamp': timestamps})

sigmoid = lambda x: 1/(1 + np.exp(-x))
rating_similarity = {1: sigmoid(-4.), 2: sigmoid(-2.), 3: sigmoid(0.), 4: sigmoid(2.), 5: sigmoid(4.)}
rating_df['rating']=rating_df['rating'].apply(lambda x: rating_similarity[int(x)])
rating_df.drop(columns=['timestamp'], inplace=True)
rating_df['user_id'] = rating_df['user_id'].astype(int)-1
rating_df['item_id'] = rating_df['item_id'].astype(int)-1
user = tf.gather(users, rating_df.user_id.values)
item = tf.gather(items, rating_df.item_id.values)

user_input = Input(shape=(818,))
item_input = Input(shape=(6166,))
user_feat = user_encoder(user_input)
user_encoder.trainable = False
# reordering layer
user_feat = layers.Dense(32)(user_feat)
item_feat = item_encoder(item_input)
item_encoder.trainable = False
# reordering layer
item_feat = layers.Dense(32)(item_feat)
dot_similarity = tf.keras.layers.Dot(axes=1)([user_feat, item_feat])
outputs =  tf.keras.activations.sigmoid(dot_similarity)

model = Model(inputs=[user_input, item_input], outputs=outputs)

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=200,
    decay_rate=0.96,
    staircase=True)

model.load_weights('./checkpoints/score_model')
new_model = Model(inputs=[user_input, item_input], outputs=model.layers[-2].output)

new_model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
	metrics=["mse"])

set_seeds()
history = new_model.fit(
    [user, item], tf.convert_to_tensor(rating_df['rating'].values),
    batch_size=512, 
    epochs=2, shuffle=True)
user_encoder.trainable = True
item_encoder.trainable = True
new_model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
	metrics=["mse"])
history = new_model.fit(
    [user, item], tf.convert_to_tensor(rating_df['rating'].values),
    batch_size=512, 
    epochs=100, shuffle=True)

new_model.save_weights('./checkpoints/score_model_cross2')
user_encoder = keras.Model(user_input, user_feat)
item_encoder = keras.Model(item_input, item_feat)
user_encoder.save_weights('./checkpoints/user_encoder_cross2')
item_encoder.save_weights('./checkpoints/item_encoder_cross2')

with open('user2.npy', 'wb') as f:
    np.save(f, user_encoder(users).numpy())

with open('item2.npy', 'wb') as f:
    np.save(f, item_encoder(items).numpy())

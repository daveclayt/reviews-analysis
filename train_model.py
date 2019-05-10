"""This script takes the clean dataframe and trains an LSTM neural network.
Needs significant computing power to run in a reasonable timeframe!"""
import read_clean_data
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import (LSTM, Activation, BatchNormalization,
                                     Dense, Embedding, Flatten)
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

json_path = "model_test.json"
h5_path = "model_test.h5"

def train_and_dump_model(vocabulary, embedding_weights, X, y):
    clear_session()

    model = Sequential()
    model.add(Embedding(len(vocabulary), 15, input_length=30, weights=[embedding_weights], trainable=False))
    model.add(LSTM(20, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(10))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, batch_size=500, epochs=100, validation_split=0.2)


    # save model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # save weights to HDF5
    model.save_weights(h5_path)
    print("Saved model to disk")

if __name__ == '__main__':
    vocab, ew, X, y = read_clean_data.main(read_clean_data.FILE)
    train_and_dump_model(vocab, ew, X, y)
    print(f'Successfully dumped model to {json_path} and weights to {h5_path}!')

from tacape.logo import logo
from tacape.utils.load import load_data
from tensorflow import keras
from tacape.models import build_model
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import tensorflow as tf
import pickle

def main():

    print(logo)

    argument_parser = ArgumentParser(prog='TACaPe: Model Training')
    argument_parser.add_argument('--positive-train', required=True, help='Input file containing positive peptides for training')
    argument_parser.add_argument('--positive-test', required=True, help='Input file containing positive peptides for testing')
    argument_parser.add_argument('--format', choices=['text', 'fasta'], default='text', help='[optional] Input file format (default: text)')
    argument_parser.add_argument('--output', required=True, help='Path prefix of the output files containing the trained model')
    argument_parser.add_argument('--epochs', type=int, default=30, help="[optional] Number of epochs to be used during training (default: 30)")
    argument_parser.add_argument('--max-vocab-size', required=False, help='[optional] Size of the vocabulary', default=24, type=int)

    arguments = argument_parser.parse_args()

    train_model(
        arguments.positive_train, 
        arguments.positive_test, 
        format=arguments.format, 
        output=arguments.output,
        epochs=arguments.epochs,
        max_vocab_size=arguments.max_vocab_size
    )

def train_model(positive_train, positive_test, output, max_vocab_size, format='text', epochs=30):

    raw_X_train = [*load_data(positive_train, format)]
    raw_X_test  = [*load_data(positive_test, format)]

    X_train = []
    y_train = []

    for sequence in raw_X_train:
        sequence = '$' + sequence + '.'
        for i in range(1, len(sequence)):
            X_train.append(sequence[0:i])
            y_train.append(sequence[i])
            print(y_train[-1])

    X_test = []
    y_test = []

    for sequence in raw_X_test:
        sequence = '$' + sequence + '.'
        for i in range(1, len(sequence)):
            X_test.append(sequence[0:i])
            y_test.append(sequence[i])

    tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
    tokenizer.fit_on_texts(X_train)

    X_train_tokens = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = 60)
    X_test_tokens  = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = 60)

    with open(output + '.tokenizer', 'wb') as writer:
        writer.write(pickle.dumps(tokenizer))

    le = LabelEncoder()
    le.fit(y_train)

    y_train_encode = keras.utils.to_categorical(le.transform(y_train))
    y_test_encode  = keras.utils.to_categorical(le.transform(y_test))

    with open(output + '.le', 'wb') as writer:
        writer.write(pickle.dumps(le))

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )]

    model = build_model(output_shape=(len(le.classes_)), embed_dim=max_vocab_size)
    model.fit(X_train_tokens, y_train_encode, validation_data=(X_test_tokens, y_test_encode), epochs=epochs, callbacks=callbacks)
    model.save_weights(output + '.weights')

if __name__ == '__main__':
    main()
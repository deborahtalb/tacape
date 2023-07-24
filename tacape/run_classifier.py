from tacape.logo import logo
from tacape.utils.load import load_data
from tensorflow import keras
from models import build_model
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

def main():

    print(logo)

    argument_parser = ArgumentParser(prog='TACaPe: Predict')
    argument_parser.add_argument('--input', required=True, help='Input file')
    argument_parser.add_argument('--format', choices=['text', 'fasta'], default='text', help='[optional] Input file format (default: text)')
    argument_parser.add_argument('--classifier-prefix', required=True, help='[optional] Path to the file prefix of the trained classification model')
    argument_parser.add_argument('--output', required=True, help='Path to the output CSV file')
    argument_parser.add_argument('--max-vocab-size', required=False, help='', default=24)

    arguments = argument_parser.parse_args()

    run_model(arguments.input, arguments.format, arguments.classifier_prefix, arguments.output, arguments.max_vocab_size)

def run_model(input, format, model_prefix, output, max_vocab_size):

    with open(model_prefix + '.tokenizer', 'rb') as reader:
        tokenizer = pickle.loads(reader.read())

    with open(model_prefix + '.le', 'rb') as reader:
        le = pickle.loads(reader.read())

    X = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(load_data(input, format)), maxlen = 60)

    model  = build_model(output_shape=(len(le.classes_)), vocab_size=max_vocab_size)
    model.load_weights(model_prefix + '.weights')
    y_pred = model.predict(X)[:,1]

    print(y_pred)

    df_output = pd.DataFrame()
    df_output['sequence'] = load_data(input, format)
    df_output['result'] = y_pred
    df_output.to_csv(output, index=False)

if __name__ == '__main__':
    main()
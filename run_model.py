from secondary_predictor import SecPredictor
from dataframe_generator import Generator
import pandas as pd
import numpy as np
import argparse

def change_label(label):
    """change the label of each class to do the next step, score Q3, Q8 and SOV"""
    res = None
    if label == 'none':
        res = '-'
    elif label == 'alpha-helix':
        res = 'H'
    elif label == 'pi-helix':
        res = 'I'
    elif label == '310-helix':
        res = 'G'
    elif label == 'parallel-sheet':
        res = 'X'
    elif label == 'antiparallel-sheet':
        res = 'X'
    return res


def main(train, test):
    # generate data frame for both training and test data
    gtrain = Generator(train, 'train')
    gtrain.read_files()
    gtest = Generator(test, 'test')
    gtest.read_files()
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_inputs = train_df[
        ['sheet_score', 'helix_score', 'tau_angle', 'theta_angle', '(i, i+3)', '(i, i+3)r', '(i, i+4)', '(i, i+4)r',
         '(i, i+5)', '(i, i+5)r']]
    train_targets = train_df['subset']
    test_inputs = test_df[
        ['sheet_score', 'helix_score', 'tau_angle', 'theta_angle', '(i, i+3)', '(i, i+3)r', '(i, i+4)', '(i, i+4)r',
         '(i, i+5)', '(i, i+5)r']]
    test_targets = test_df['subset']

    # fit model
    p = SecPredictor()
    p.fit(train_inputs, train_targets)
    pre = p.predict(test_inputs)
    # predict test data
    comp_df = pd.DataFrame({'predicted': pre, 'observed': np.array(test_targets)})
    comp_df.to_csv('predicted_table.csv')
    comp_df = comp_df.applymap(change_label)
    # write a compare table for test data. (test protein can be change)
    comp_df.to_csv('compare_table.csv')
    print('\nProcesses end.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run secondary prediction")
    parser.add_argument("train_folder", help="Input folder path includes all training pdb files")
    parser.add_argument("test_folder", help="Input folder path includes all test pdb files")
    args = parser.parse_args()

    main(args.train_folder, args.test_folder)

from secondary_predictor import SecPredictor
import pandas as pd
import argparse
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV


def main(train):
    # read training data table
    print('Reading data...')
    df = pd.read_csv(train)
    inputs = df[['sheet_score', 'helix_score', 'tau_angle', 'theta_angle', '(i, i+3)',
                 '(i, i+3)r', '(i, i+4)', '(i, i+4)r', '(i, i+5)', '(i, i+5)r']]
    targets = df['subset']
    print('Reading end.\n')

    # parameters optimization
    #print('Begin parameters optimization...')
    #param_grid = {'pca_n': [6, 7, 8, 9, 10], 'svm_k': ['rbf', 'poly', 'linear']}
    #grid_search = GridSearchCV(SecPredictor(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True))
    #grid_search.fit(inputs, targets)
    #print(grid_search.best_score_)
    #print(grid_search.best_params_)

    # cross validation by using "cross_val_score" function
    print('Begin 4-fold cross validation...')
    cv_scores = cross_val_score(SecPredictor(), inputs, targets,
                                cv=StratifiedKFold(n_splits=4, shuffle=True))
    print('CV scores: ', cv_scores)
    print('CV scores mean:', cv_scores.mean(), 'stdev:', cv_scores.std())
    print('End cross validation.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross_validation")
    parser.add_argument("train_dataframe", help="Input training.csv")
    args = parser.parse_args()
    main(args.train_dataframe)
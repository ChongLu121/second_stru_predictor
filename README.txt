Please run the program in the folder including all 4 .py files

1. generate data frame (csv file) from pdb files:
$ python3 dataframe_generator.py [data folder] [output .csv file name]

2. 4-fold cross validation:
$ python3 po_cv.py [.csv file path]

3. secondary structure annotation:
$ python3 run_model.py [training data folder path] [test data folder path]


To do the cross validation, please generate the data frame first.
To do the annotation, please input the third command line.

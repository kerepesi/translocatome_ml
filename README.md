
**************************************************************************
Tables and codes for machine learning prediction of tranclocating proteins
**************************************************************************

Requirements (versions used by us in parenthesis):

   - Linux (Ubuntu 16.04.3 LTS)
   - Python 3.5.2
   - Python packages: pandas (0.20.3), numpy (1.13.1), sklearn (0.19.0), xgboost (0.6)

Running commands:

  - Running of 100 experiments of XGBoost with 5 fold CV (predictions are averaged, parameters: max_depth=1, n_est=80, lr=0.3, n_exp=1
    $ python XGBoost_CV.py 0 GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv

  - Dumping trees of the XGBoost model (max_d=1, n_est=80, lr=0.3):
    $ python Tree.py GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv

  - Creating a simple model (rule table):
    $ python RuleTable.py GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv

Description of the files:

  - translocatome_labels_posneg_add.csv
    Labels of the classification ('1' if the given protein included is a translocating proteins, 0 otherwise)

  - GO_features.csv:
    Gene Ontology features for the training proteins

  - GO_features.csv-imp-n_est80-list.csv:
    Important features selected by XGBoost (n_est=80, threshold = 0)

  - GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv:
    Table of the final 17 features (selected by XGBoost with the parameter setting n_est=80, threshold=0.02)

  - GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv_Trees-n_est80-max_d1.txt:
    Output file of the command 'python Tree.py GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv'
    
  - GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv_Trees-n_est80-max_d1.txt-RuleTable.py.csv
    Output file of the command 'python RuleTable.py GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv_Trees-n_est80-max_d1.txt'

  - GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv-XGBoost_CV_preds-n_est80-exp100.csv:
    Output file of the command 'python XGBoost_CV.py 1 GO_features.csv-imp-n_est80-thr0.02-table.csv-add_degree_bridgeness.csv'

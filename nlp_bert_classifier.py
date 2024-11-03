# Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast, BertModel
import torch
from scipy.stats import uniform

# Class Imports
from helpful_functions import get_device

# Define Device
device = get_device()

# Defining file paths
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
submission_df = pd.read_csv('./data/sample_submission.csv')

# Initialize Tokenizer And Bert Model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased').to(device)
model.eval() # Set To Eval Mode




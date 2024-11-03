# Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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


def get_bert_embeddings(texts, layer_aggregation='last_4_mean', batch_size=16):
    """Get BERT embeddings for a list of texts in batches, using the specified device."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(**encoded_input)

        if layer_aggregation == 'last':
            embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
        elif layer_aggregation == 'last_4_mean':
            embeddings = torch.mean(torch.stack(output.hidden_states[-4:]), dim=0).mean(dim=1).cpu().numpy()

        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Generate BERT embeddings for train and test datasets
train_embeddings = get_bert_embeddings(train_df["text"].tolist(), layer_aggregation='last_4_mean')
test_embeddings = get_bert_embeddings(test_df["text"].tolist(), layer_aggregation='last_4_mean')

# Scale embeddings
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Define the RidgeClassifier model and parameter distribution
ridge_model = RidgeClassifier()
param_distributions = {
    'alpha': uniform(0.001, 100),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}

# Use StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(ridge_model, param_distributions, n_iter=30, scoring='f1', cv=kf, n_jobs=-1, random_state=42)
random_search.fit(train_embeddings, train_df["target"])

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_
print(f"Best parameters: {best_params} with F1 Score: {random_search.best_score_}")

# Predict on the test set with the best model
test_predictions = best_model.predict(test_embeddings)

# Prepare the submission file
submission_df["target"] = test_predictions

# Save the submission file
submission_df.to_csv("./data/submission.csv", index=False)




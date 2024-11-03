****BERT Text Classification with Ridge Classifier****

This project is a text classification model utilizing BERT embeddings for feature extraction and a Ridge Classifier to predict target labels. The BERT model is used to generate text embeddings, which are then fed into a Ridge Classifier with tuned hyperparameters to achieve the best classification performance.

****Project Structure****

	1.	Feature Extraction: BERT embeddings are generated for each text input, utilizing the bert-base-uncased model. The embeddings are computed by taking the mean of the last hidden state from the BERT model.
	2.	Modeling: A Ridge Classifier is used with cross-validation to find the optimal regularization strength. Hyperparameter tuning is conducted to select the best model based on the F1 score.
	3.	Prediction and Submission: The model with the best hyperparameters is used to generate predictions on the test data. The results are saved in a format suitable for submission.

****Requirements****

The following Python libraries are required to run the project:

	•	numpy
	•	pandas
	•	scikit-learn
	•	torch
	•	transformers

Make sure these dependencies are included in your requirements.txt file.

****Installation****

	1.	Clone the repository:

git clone <repository-url>
cd <repository-folder>


	2.	Install Requirements:
Use the requirements.txt file to install all necessary libraries:

pip install -r requirements.txt


****Usage****

	1.	Load Data: The code loads training and test data from ./data/train.csv and ./data/test.csv.
	2.	Generate Embeddings: BERT embeddings are generated for each text sample using the get_bert_embeddings function.
	3.	Hyperparameter Tuning: The Ridge Classifier is trained with different values of alpha to find the best F1 score using cross-validation.
	4.	Prediction: The best model is used to predict on the test dataset, and the results are saved as ./data/submission.csv.

****Code Structure****

	•	BERT Embeddings: Text data is transformed into embeddings using the get_bert_embeddings function, which leverages the BERT model.
	•	Cross-Validation: A KFold cross-validator is used to evaluate the Ridge Classifier with different values of alpha.
	•	Best Model Selection: The model with the highest average F1 score during cross-validation is selected.
	•	Submission File: The predictions on the test data are saved in a CSV file in the format required for submission.

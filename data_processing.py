import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import csv

import pandas as pd

model_rf = None  # Global variable for the Random Forest model
model_xgb = None  # Global variable for the XGBoost model
vectorizer=None
df2=None
scaler = None
def train_model():
    global model_rf, model_xgb,vectorizer,df2,scaler

    df2 = pd.read_csv('seq.csv')
    X = df2['Sequence']
    y = df2['Subpopulation']

    # Convert input sequences to numerical features using CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X_kmers = vectorizer.fit_transform(X)
    encoded_1 = X_kmers.toarray()
    X = pd.DataFrame(encoded_1)
    x_encode = X
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Random Forest Classifier model
    model_rf = RandomForestClassifier(n_estimators=30, min_samples_leaf=1, min_samples_split=2,
                                      max_features=0.3010, max_depth=10)
    model_rf.fit(X_train, y_train)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Concatenate X_encode and y_encoded
    X1 = np.concatenate((x_encode, y_encoded.reshape(-1, 1)), axis=1)
    y1 = df2["Plant Height (cm)"]

    # Split the data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)

    # Define the XGBoost model
    model_xgb = xgb.XGBRegressor()

    # Set the parameter grid for hyperparameter tuning
    param_grid = {
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [10, 20, 30],
        'n_estimators': [50, 100, 150]
    }

    # Perform grid search with cross-validation on the training data
    grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5)
    grid_search.fit(X1_train, y1_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    model_xgb = grid_search.best_estimator_

    # Fit the best model on the entire dataset
    model_xgb.fit(X1, y1)
    y_pred1=model_xgb.predict(X1_test)
    print(mean_squared_error(y1_test,y_pred1))

def predict_height(input_sequence):
    global model_rf, model_xgb
#new_input_df=new_sequence_X 
#
    # Convert the new input sequence to numerical features
    new_sequence_kmers = vectorizer.transform([input_sequence])
    new_sequence_encoded = new_sequence_kmers.toarray()
    new_sequence_X = pd.DataFrame(new_sequence_encoded)

    # Scale the features of the new sequence
    
    # new_sequence_scaled = scaler.fit_transform(new_sequence_X)
    new_input_df_scaled=scaler.transform(new_sequence_X)

    # Make prediction on the new sequence using the Random Forest model
    #predicted_subpopulation = model_rf.predict(new_sequence_scaled)
    predicted_subpopulation=model_rf.predict(new_input_df_scaled)

    # Make prediction on the new sequence using the XGBoost model
    le = LabelEncoder()
    y = df2['Subpopulation']
    le.fit(y)

    # Encode the new subpopulation
    new_subpopulation_encoded = le.transform([predicted_subpopulation[0]])

    # Concatenate the encoded new sequence and subpopulation
    new_data_encoded = np.concatenate((new_input_df_scaled, new_subpopulation_encoded.reshape(-1, 1)), axis=1)
    predicted_height = model_xgb.predict(new_data_encoded)[0]

    return predicted_subpopulation[0], predicted_height
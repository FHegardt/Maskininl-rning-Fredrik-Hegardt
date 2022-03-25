import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


myModel = joblib.load("./Labb./best_model.pkl")
test_data = pd.read_csv("./Labb./test_samples.csv", index_col=0)

probability = myModel.predict_proba(test_data)
prediction = myModel.predict(test_data)
predicted_data = pd.DataFrame(probability,prediction)
predicted_data.to_csv("./Labb/predictions.csv",  index_label= ["Prediction", "Probability class 0", "Probability class 1"])
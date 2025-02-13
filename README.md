# Crop Recommendation System

This is a Flask-based web application that recommends the best crop to cultivate based on soil and environmental parameters.

## Features
- Accepts input parameters: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
- Utilizes a trained Random Forest Classifier for predictions.
- Normalizes input data using MinMaxScaler and StandardScaler.
- Returns the best crop recommendation based on input data.

## Dataset
The model is trained on the `Crop_recommendation.csv` dataset, which includes:
- Soil macronutrients (N, P, K)
- Climate conditions (Temperature, Humidity, Rainfall)
- Soil pH levels
- Crop labels

## File Structure
```
├── app.py               # Flask application
├── templates/
│   ├── index.html       # Frontend UI
├── Crop_recommendation.csv  # Dataset
```

## Dependencies
- Flask
- Pandas
- NumPy
- Scikit-learn

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.


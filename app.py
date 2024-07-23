# importing required functions for flask app and model
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and preprocess objects
with open('recommend_model.pkl', 'rb') as f:
    kmeans_main_model, scaler_main, label_encoders = pickle.load(f)

# Load the encoded dataset along with finalized clusters
df = pd.read_csv('encoded_csv.csv')

# Selected columns for recommendation feature
selected_columns = ['Type', 'BodyPart', 'Level', 'Equipment']
# Re implementing label encoders and fitting the dataset
for column in selected_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Standardize the data again to ensure consistency
scaled_data = scaler_main.transform(df[selected_columns])
df['Cluster_main'] = kmeans_main_model.predict(scaled_data)


def recommend_items_by_attributes(attributes_values, data, label_encoders, scaler, kmeans, selected_columns,
                                  n_recommendations=5):
    attribute_vector = []
    for col in selected_columns:
        try:
            attribute_vector.append(label_encoders[col].transform([attributes_values[col]])[0])
        except ValueError as e:
            print(f"Unseen label detected in {col}. Using the default value: 0")
            attribute_vector.append(0)  # Default value if unseen label detected

    attribute_vector = np.array(attribute_vector).reshape(1, -1)
    attribute_vector_df = pd.DataFrame(attribute_vector, columns=selected_columns)
    attribute_vector_scaled = scaler.transform(attribute_vector_df)
    cluster_label = kmeans.predict(attribute_vector_scaled)[0]
    cluster_data = data[data['Cluster_main'] == cluster_label]
    n_recommendations = min(len(cluster_data), n_recommendations)
    recommendations = cluster_data.sample(n=n_recommendations).index.tolist()
    return recommendations

# Index/Home page rendering
@app.route('/')
def home():
    return render_template('index.html')

# Recommendation page rendering
@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.form
    try:
        attributes_values = {}
        for col in ['Type', 'BodyPart', 'Level', 'Equipment']:
            try:
                input_value = content[col]
                print(f"Input value for {col}: {input_value}")
                print(f"Possible values for {col}: {list(label_encoders[col].classes_)}")
                default_value = df[col].mode()[0]  # Use the most frequent value in case of unseen label
                attributes_values[col] = label_encoders[col].transform([input_value])[0]

            except ValueError as e:
                # Handle unseen label by assigning the most frequent category or a default category
                print(f"Error: {str(e)}")
                attributes_values[col] = df[col].mode()[0]
                print(f"Unseen label detected in {col}. Using the default value: {attributes_values[col]}")
    except KeyError as e:
        return render_template('index.html', error=f"Invalid attribute: {str(e)}")

    # Make recommendations calling the function
    recommendations = recommend_items_by_attributes(attributes_values, df, label_encoders, scaler_main,kmeans_main_model,
                                                    selected_columns, n_recommendations=9)
    # Read non encoded dataset
    result_df = pd.read_csv('cleaned_data.csv')
    print(recommendations)
    # Store the recommended values from dataset and send it to the recommendationn page
    recommended_items = result_df.iloc[recommendations].to_dict('records')
    return render_template('recommendations.html', recommendations=recommended_items)


if __name__ == '__main__':
    app.run(debug=True)

    """
    
    Input value for Type: 4 Possible values for Type: ['Cardio', 'Olympic Weightlifting', 'Plyometrics', 
    'Powerlifting', 'Strength', 'Stretching', 'Strongman'] Error: y contains previously unseen labels: 4


Unseen label detected in Type. Using the default value: 4 Input value for BodyPart: 0 Possible values for BodyPart: [
'Abdominals', 'Abductors', 'Adductors', 'Biceps', 'Calves', 'Chest', 'Forearms', 'Glutes', 'Hamstrings', 'Lats', 
'Lower Back', 'Middle Back', 'Neck', 'Quadriceps', 'Shoulders', 'Traps', 'Triceps'] Error: y contains previously 
unseen labels: 0 Unseen label detected in BodyPart. Using the default value: 0


Input value for Level: 2
Possible values for Level: ['Beginner', 'Expert', 'Intermediate']
Error: y contains previously unseen labels: 2
Unseen label detected in Level. Using the default value: 2


Input value for Equipment: 4 Possible values for Equipment: ['Bands', 'Barbell', 'Body Only', 'Cable', 
'Data Unavailable', 'Dumbbell', 'E-Z Curl Bar', 'Exercise Ball', 'Foam Roll', 'Kettlebells', 'Machine', 
'Medicine Ball', 'Other'] Error: y contains previously unseen labels: 4 Unseen label detected in Equipment. Using the 
default value: 2
    
    """

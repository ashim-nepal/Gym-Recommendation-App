# importing required functions for flask app and model
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import random


app = Flask(__name__)

# Load the model and preprocess objects
try:
    with open('recommendation_model.pkl', 'rb') as f:
        kmodes_main_model, scaler_main,  label_encoder = pickle.load(f)
        print('loaded')
except Exception as e:
    print(e)

# Load the encoded dataset along with finalized clusters
df = pd.read_csv('encoded_csv.csv')

# Selected columns for recommendation feature
selected_columns = ['BodyPart', 'Equipment']
# Re implementing label encoders and fitting the dataset

for column in selected_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoder[column] = le

# Standardize the data again to ensure consistency
scaled_data = scaler_main.transform(df[selected_columns])
df['Cluster_main'] = kmodes_main_model.predict(scaled_data)

def recommend_items(user_features, df, top_n):
    # Create DataFrame for user features
    user_df = pd.DataFrame([user_features])

    # Calculate similarity between user_features and all items
    item_features = df[['BodyPart', 'Equipment']]
    user_feature_array = user_df[['BodyPart', 'Equipment']].values
    item_feature_array = item_features.values

    # Check if item_feature_array has elements
    if item_feature_array.size == 0:
        raise ValueError("Item feature array is empty. Check the input data.")

    similarities = cosine_similarity(user_feature_array, item_feature_array)

    # Check if similarities result in empty array
    if similarities.size == 0:
        raise ValueError("Similarities calculation returned an empty result. Check the input data.")

    # Find the most similar item
    closest_item_index = similarities[0].argmax()
    closest_item = df.iloc[closest_item_index]

    user_cluster = closest_item['Cluster_main']

    # Filter items by cluster
    cluster_items = df[df['Cluster_main'] == user_cluster]

    if cluster_items.empty:
        raise ValueError(f"No items found in the cluster {user_cluster}.")

    # Compute similarity between the user features and the items in the same cluster
    item_features = cluster_items[['BodyPart', 'Equipment']]
    item_feature_array = item_features.values

    # Check if item_feature_array has elements
    if item_feature_array.size == 0:
        raise ValueError("Cluster item feature array is empty. Check the input data.")

    similarities = cosine_similarity(user_feature_array, item_feature_array)

    if similarities.size == 0:
        raise ValueError("Similarities calculation for cluster items returned an empty result.")

    # Get top_n most similar items
    similar_indices = similarities[0].argsort()[-top_n:][::-1]
    recommended_items = cluster_items.iloc[similar_indices]

    return recommended_items.index.tolist() #[['item_id']]

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
        for col in ['BodyPart', 'Equipment']:
            try:
                input_value = content[col]
                print(f"Input value for {col}: {input_value}")
                print(f"Possible values for {col}: {list(label_encoder[col].classes_)}")
                default_value = df[col].mode()[0]  # Use the most frequent value in case of unseen label
                attributes_values[col] = label_encoder[col].transform([input_value])[0]

            except ValueError as e:
                # Handle unseen label by assigning the most frequent category or a default category
                print(f"Error: {str(e)}")
                attributes_values[col] = df[col].mode()[0]
                print(f"Unseen label detected in {col}. Using the default value: {attributes_values[col]}")
    except KeyError as e:
        return render_template('index.html', error=f"Invalid attribute: {str(e)}")

    num_recommend = random.randint(8,18)
    print(num_recommend)
    recommendations = recommend_items(attributes_values, df, num_recommend)
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

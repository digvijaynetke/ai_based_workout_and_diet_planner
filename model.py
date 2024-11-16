from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
file_path = "C:/Users/digvi/OneDrive/Desktop/AI_Diet_Project/Project/Merged file.csv"
data = pd.read_csv(file_path)

# Preprocess the dataset
data.columns = data.columns.str.strip()
data['Weight'] = pd.to_numeric(data['Weight'], errors='coerce')
data['Height'] = pd.to_numeric(data['Height'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data.dropna(subset=['Weight', 'Height', 'Age'])

# Features and targets
features = ['Weight', 'Height', 'Age', 'Gender', 'Diet Type', 'Workout Type', 'Goal', 'Health Issues', 'Physical Activity Level']
target_diet = 'Recommended Diet Plan'
target_workout = 'Recommended Workout Routine'

# Split the data into training and test sets
X = data[features]
y_diet = data[target_diet]
y_workout = data[target_workout]

X_train, X_test, y_train_diet, y_test_diet = train_test_split(X, y_diet, test_size=0.2, random_state=42)
_, _, y_train_workout, y_test_workout = train_test_split(X, y_workout, test_size=0.2, random_state=42)

# Preprocessing: numerical and categorical features
numeric_features = ['Weight', 'Height', 'Age']
categorical_features = ['Gender', 'Diet Type', 'Workout Type', 'Goal', 'Health Issues', 'Physical Activity Level']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Preprocess and fit models
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Train models for diet and workout recommendations
model_diet = RandomForestClassifier(random_state=42)
model_diet.fit(X_train_preprocessed, y_train_diet)

model_workout = RandomForestClassifier(random_state=42)
model_workout.fit(X_train_preprocessed, y_train_workout)


def create_graph(title, labels, values):
    fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size
    # Ensure there's a small value for non-selected options to display all bars
    values = [v if v > 0 else 0.1 for v in values]
    
    ax.bar(labels, values, color=['#4CAF50', '#2196F3', '#FF5733', '#FFC300'])
    ax.set_title(title)
    ax.set_ylabel('Recommendation Score')
    ax.set_xlabel('Options')
    ax.set_ylim(0, 1.5)  # Set y-axis limit for better visibility
    
    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Save the plot tightly around content
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Route for home page
@app.route("/")
def welcome():
    return render_template("index.html")

# Routes for other pages
@app.route("/doctor.html")
def doctor():
    return render_template("doctor.html")

@app.route("/gymtrainer.html")
def gymtrainer():
    return render_template("gymtrainer.html")

@app.route("/dietian.html")
def dietian():
    return render_template("dietian.html")

@app.route("/shopping.html")
def shopping():
    return render_template("shopping.html")

# Route to handle form input and predictions
@app.route("/diet_form.html", methods=["GET", "POST"])
def diet_form():
    if request.method == "POST":
        # Handle form submission (POST request)
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        age = float(request.form["age"])
        gender = request.form["gender"]
        diet_type = request.form["diet-type"]
        workout_type = request.form["workout-type"]
        goal = request.form["goal"]
        health_issues = request.form["health-issues"]
        physical_activity = request.form["physical-activity"]

        # Create DataFrame for the input and preprocess
        input_data = pd.DataFrame({
            'Weight': [weight],
            'Height': [height],
            'Age': [age],
            'Gender': [gender],
            'Diet Type': [diet_type],
            'Workout Type': [workout_type],
            'Goal': [goal],
            'Health Issues': [health_issues],
            'Physical Activity Level': [physical_activity]
        })
        input_preprocessed = preprocessor.transform(input_data)

        # Make predictions for diet and workout
        recommended_diet = model_diet.predict(input_preprocessed)[0]
        recommended_workout = model_workout.predict(input_preprocessed)[0]

        # Redirect to results page with the recommendations
        return redirect(url_for('results', diet=recommended_diet, workout=recommended_workout))
    
    # If it's a GET request, render the form
    return render_template("diet_form.html")

# Route for displaying the recommendations and generating graphs
@app.route("/results")
def results():
    # Get the recommendations from the query string
    recommended_diet = request.args.get('diet')
    recommended_workout = request.args.get('workout')

    # Example categories for the graphs
    diet_labels = ['Low Carb', 'High Protein', 'Balanced', 'Keto']
    workout_labels = ['Cardio', 'Strength', 'Flexibility', 'HIIT']

    # Set values based on the recommendations
    diet_values = [1 if label == recommended_diet else 0 for label in diet_labels]
    workout_values = [1 if label == recommended_workout else 0 for label in workout_labels]

    # Create graphs for diet and workout recommendations
    diet_graph = create_graph('Diet Recommendation', diet_labels, diet_values)
    workout_graph = create_graph('Workout Recommendation', workout_labels, workout_values)

    # Render the results template with the recommendations and graphs
    return render_template("results.html", 
                           recommended_diet=recommended_diet, 
                           recommended_workout=recommended_workout,
                           diet_graph=diet_graph,
                           workout_graph=workout_graph)

if __name__ == "__main__":
    app.run(debug=True) 
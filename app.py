from flask import Flask, render_template, request, session
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load pre-trained KNN model and label encoders
model = joblib.load('model/rf_model.pkl')  # Load the KNN model
label_encoders = joblib.load('model/label_encoders.pkl')  # Load label encoders

# Route for the main form (index page)
@app.route('/')
def index():
    return render_template('index.html')

# Route for vessel details page
@app.route('/vessel_details', methods=['POST'])
def vessel_details():
    session['boat_type'] = request.form['boat_type']
    session['model'] = request.form['model']
    session['manufacturer'] = request.form['manufacturer']
    session['vessel_age'] = 2024 - int(request.form['year_built'])  # Calculate vessel age based on current year
    return render_template('vessel_details.html')

# Route for engine details page
@app.route('/engine_details', methods=['POST'])
def engine_details():
    session['displacement'] = request.form['displacement']
    session['length_overall'] = request.form['length_overall']
    session['beam'] = request.form['beam']
    session['hull_material'] = request.form['hull_material']
    session['maintenance_frequency'] = request.form['maintenance_frequency']
    return render_template('engine_details.html')

# Route for total value page
@app.route('/total_value', methods=['POST'])
def total_value():
    session['engine_type'] = request.form['engine_type']
    session['number_of_engines'] = request.form['number_of_engines']
    session['fuel_type'] = request.form['fuel_type']
    session['engine_power'] = request.form['engine_power']
    session['engine_manufacturer'] = request.form['engine_manufacturer']
    session['total_engine_power'] = request.form['total_engine_power']
    session['safety_features'] = request.form['safety_features']
    return render_template('total_value.html')

# Route for cruising area page
@app.route('/cruising_area', methods=['POST'])
def cruising_area():
    session['total_insured_value'] = request.form['total_insured_value']
    session['personal_effects_value'] = request.form['personal_effects_value']
    session['insurance_type'] = request.form['insurance_type']
    session['currency'] = request.form['currency']
    session['payment_option'] = request.form['payment_option']
    return render_template('cruising_area.html')

# Route for results page
@app.route('/results', methods=['POST'])
def results():
    session['country_of_residence'] = request.form['country_of_residence']
    session['registered_country'] = request.form['registered_country']
    session['usage'] = request.form['usage']
    session['summer_mooring'] = request.form['summer_mooring']
    session['winter_mooring'] = request.form['winter_mooring']
    session['cruising_areas'] = request.form.getlist('cruising_areas')
    session['boating_experience'] = request.form['boating_experience']
    session['mileage_experience'] = request.form['mileage_experience']

    # Prepare input data for the ML model
    input_data = pd.DataFrame([[
        session['vessel_age'], 
        session['hull_material'], 
        session['engine_power'], 
        session['fuel_type'], 
        session['displacement'], 
        session['cruising_areas'][0], 
        session['boat_type'], 
        session['number_of_engines'], 
        session['personal_effects_value'], 
        session['insurance_type'], 
        session['summer_mooring'], 
        session['winter_mooring'], 
        session['boating_experience'], 
        session['mileage_experience'], 
        session['maintenance_frequency'], 
        session['safety_features']
    ]], columns=[
        'vessel_age', 'hull_material', 'engine_power', 'fuel_type', 'displacement', 'cruising_area', 
        'boat_type', 'number_of_engines', 'personal_effects_value', 'insurance_type', 'summer_mooring', 
        'winter_mooring', 'years_boating_experience', 'mileage_experience', 'maintenance_frequency', 
        'safety_features'
    ])

    # Preprocess data using the label encoders
    input_data = preprocess_data(input_data)

    # Predict the risk score using KNN's probability prediction (likelihood of a claim)
    risk_score = model.predict_proba(input_data)[0][1] * 100  # Probability of 'Yes' claim

    # Calculate the insurance quote based on a simple formula
    quote = calculate_quote(session['boat_type'], session['total_insured_value'], session['insurance_type'])

    return render_template('results.html', risk_score=round(risk_score, 2), quote=quote)

# Function to preprocess the input data using label encoders
def preprocess_data(input_data):
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])
    return input_data

# Function to calculate insurance quote (simple calculation)
def calculate_quote(boat_type, total_insured_value, insurance_type):
    base_rate = 0.05  # 5% Base rate for insurance

    # Adjust the rate based on the boat type
    if boat_type == 'Cargo':
        rate_modifier = 1.02  # 2% increase for Cargo
    elif boat_type == 'Yacht':
        rate_modifier = 1.03  # 3% increase for Yacht
    else:
        rate_modifier = 1  # No additional rate for other types

    # Adjust the rate based on the insurance type
    if insurance_type == 'Hull and Third Party Liability':
        insurance_modifier = 1.05  # 5% increase for full insurance
    else:
        insurance_modifier = 1  # No increase for Third Party Liability Only

    # Calculate the final quote with additional expenses (10%)
    quote = base_rate * rate_modifier * insurance_modifier * float(total_insured_value) * 1.10
    
    return round(quote, 2)


# Start the app
if __name__ == '__main__':
    app.run(debug=True)






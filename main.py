from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#DATABASES
#load databases
precautions = pd.read_csv(r"datasets/precautions_df.csv")
workout =  pd.read_csv(r"datasets/workout_df.csv")
try:
    description = pd.read_csv("datasets/description.csv", encoding='utf-8')
except:
    description = pd.read_csv("datasets/description.csv", encoding='latin1')
medications = pd.read_csv(r"datasets/medications.csv")
diets = pd.read_csv(r"datasets/diets.csv")

#normalize disease names to avoid whitespace mismatches between csv's
if 'Disease' in precautions.columns:
    precautions['Disease'] = precautions['Disease'].astype(str).str.strip()
if 'disease' in workout.columns:
    workout['disease'] = workout['disease'].astype(str).str.strip()
elif 'Disease' in workout.columns:
    workout['Disease'] = workout['Disease'].astype(str).str.strip()

if 'Disease' in description.columns:
    description['Disease'] = description['Disease'].astype(str).str.strip()
if 'Disease' in medications.columns:
    medications['Disease'] = medications['Disease'].astype(str).str.strip()
if 'Disease' in diets.columns:
    diets['Disease'] = diets['Disease'].astype(str).str.strip()

app = Flask(__name__, static_folder='static', static_url_path='/static', template_folder='templates')


# HELPER FUNCTION
def helper(dis):
    dis = str(dis).strip()

    descr = description[description['Disease'] == dis]['Description']
    descr = " ".join([w for w in descr]) if not descr.empty else "No description available"

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values] if not pre.empty else []

    med_df = medications[medications['Disease'] == dis]['Medication']
    if not med_df.empty:
        med_str = med_df.iloc[0]  #get the string representation
        #parse the string representation of list
        import ast
        try:
            if pd.isna(med_str):
                med = []
            else:
                med = ast.literal_eval(med_str)
        except Exception:
            med = [] if pd.isna(med_str) else [str(med_str)]  #fallback if parsing fails
    else:
        med = []

    diet_df = diets[diets['Disease'] == dis]['Diet']
    if not diet_df.empty:
        diet_str = diet_df.iloc[0]  #get the string representation
        #parse the string representation of list
        import ast
        try:
            if pd.isna(diet_str):
                diet = []
            else:
                diet = ast.literal_eval(diet_str)
        except Exception:
            diet = [] if pd.isna(diet_str) else [str(diet_str)]  #fallback if parsing fails
    else:
        diet = []

    # `datasets/workout_df.csv` uses columns like workout1..workout10
    disease_col = 'disease' if 'disease' in workout.columns else 'Disease'
    workout_cols = [c for c in workout.columns if str(c).lower().startswith('workout')]
    if disease_col not in workout.columns or not workout_cols:
        wrkout = []
    else:
        rows = workout.loc[workout[disease_col] == dis, workout_cols]
        if rows.empty:
            wrkout = []
        else:
            #flatten, remove blanks/NaNs, keep as strings.
            vals = rows.values.flatten().tolist()
            wrkout = [str(v).strip() for v in vals if not pd.isna(v) and str(v).strip()]

    return descr, pre, med, diet, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131 , 'spinning_sensation' : 132 ,
                 'difficulty_swallowing' : 133, 'swollen_tonsils' :134, 'rapid_pulse' : 135, 'dry_mouth' : 136, 'night_vision_problems' : 137,
                 'faded_colors' : 138 , 'shortness_of_breath': 139, 'facial_pain': 140, 'frequent_urination': 141, 'acid_taste': 142,
                 'limited_motion': 143, 'severe_pain': 144, 'nasal_congestion': 145, 'eye_pain': 146, 'bad_breath': 147, 'blood_in_urine': 148, 'pain_at_night': 149,
                 'chronic_cough': 150, 'dry_skin': 151, 'high_body_temperature': 152, 'bloating': 153, 'thirst': 154, 'confusion': 155, 'heatburn': 156, 'leg_pain': 157,
                 'sore_throat': 158, 'weakness': 159, 'shoulder_pain': 160, 'redness': 161, 'swelling': 162, 'tingling': 163, 'stiffness': 164 , 'burping': 165,
                 'halos': 166, 'lower_back_pain': 167, 'chest_discomfort': 168, 'numbness': 169, 'sudden_pain_attacks': 170, 'wheezing': 171 ,'vision_loss' : 172 }
def _normalize_feature_name(name: str) -> str:

    import re

    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name


def train_svc_model(training_csv_path: str = r"datasets/Training.csv") -> LinearSVC:

    df = pd.read_csv(training_csv_path)

    label_col = "prognosis" if "prognosis" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    #map normalized Training-column names
    norm_to_col = {}
    for c in feature_cols:
        n = _normalize_feature_name(c)
        if n not in norm_to_col:
            norm_to_col[n] = c


    X = np.zeros((len(df), len(symptoms_dict)), dtype=np.float32)
    missing = []
    for symptom, idx in symptoms_dict.items():
        col = norm_to_col.get(_normalize_feature_name(symptom))
        if col is None:
            missing.append(symptom)
            continue
        X[:, idx] = df[col].values

    if missing:
        print(f"[Model Training] Warning: {len(missing)} symptoms not found in Training.csv. They will be left as 0.")

    y = df[label_col].astype(str).values


    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Model Training] Test accuracy: {acc:.4f}")

    return model


#training the model
svc = train_svc_model()


# MODEL PREDICTION FUNCTION
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    
    # mapping for common symptom variations
    symptom_mapping = {
        'coughing': 'cough',
        'cough': 'cough',
        'headache': 'headache',
        'head pain': 'headache',
        'fever': 'high_fever',
        'high fever': 'high_fever',
        'mild fever': 'mild_fever',
        'nausea': 'nausea',
        'nauseous': 'nausea',
        'vomiting': 'vomiting',
        'throw up': 'vomiting',
        'fatigue': 'fatigue',
        'tired': 'fatigue',
        'exhausted': 'fatigue',
        'chest pain': 'chest_pain',
        'chest ache': 'chest_pain',
        'breathing problems': 'breathlessness',
        'shortness of breath': 'breathlessness',
        'difficulty breathing': 'breathlessness',
        'stomach pain': 'stomach_pain',
        'abdominal pain': 'abdominal_pain',
        'belly pain': 'belly_pain',
        'back pain': 'back_pain',
        'joint pain': 'joint_pain',
        'muscle pain': 'muscle_pain',
        'diarrhea': 'diarrhoea',
        'diarrhoea': 'diarrhoea',
        'constipation': 'constipation',
        'skin rash': 'skin_rash',
        'rash': 'skin_rash',
        'itching': 'itching',
        'itchy': 'itching',
        'sneezing': 'continuous_sneezing',
        'runny nose': 'runny_nose',
        'stuffy nose': 'congestion',
        'congestion': 'congestion',
        'sore throat': 'throat_irritation',
        'throat pain': 'throat_irritation',
        'dizziness': 'dizziness',
        'dizzy': 'dizziness',
        'weight gain': 'weight_gain',
        'weight loss': 'weight_loss',
        'anxiety': 'anxiety',
        'anxious': 'anxiety',
        'eye pain': 'pain_behind_the_eyes',
        'depression': 'depression',
        'depressed': 'depression',
        'mood swings': 'mood_swings',
        'mood changes': 'mood_swings',
        'sweating': 'sweating',
        'sweat': 'sweating',
        'chills': 'chills',
        'shivering': 'shivering',
        'cold hands': 'cold_hands_and_feets',
        'cold feet': 'cold_hands_and_feets',
        'swollen legs': 'swollen_legs',
        'leg swelling': 'swollen_legs',
        'neck pain': 'neck_pain',
        'stiff neck': 'stiff_neck',
        'knee pain': 'knee_pain',
        'hip pain': 'hip_joint_pain',
        'muscle weakness': 'muscle_weakness',
        'weak muscles': 'muscle_weakness',
        'loss of appetite': 'loss_of_appetite',
        'no appetite': 'loss_of_appetite',
        'increased appetite': 'increased_appetite',
        'hungry': 'increased_appetite',
        'thirsty': 'polyuria',
        'frequent urination': 'polyuria',
        'painful urination': 'burning_micturition',
        'burning urination': 'burning_micturition',
        'blood in urine': 'spotting_urination',
        'dark urine': 'dark_urine',
        'yellow urine': 'yellow_urine',
        'yellow skin': 'yellowish_skin',
        'jaundice': 'yellowish_skin',
        'yellow eyes': 'yellowing_of_eyes',
        'blurred vision': 'blurred_and_distorted_vision',
        'vision problems': 'visual_disturbances',
        'eye redness': 'redness_of_eyes',
        'red eyes': 'redness_of_eyes',
        'watery eyes': 'watering_from_eyes',
        'eye watering': 'watering_from_eyes',
        'sinus pressure': 'sinus_pressure',
        'sinus pain': 'sinus_pressure',
        'phlegm': 'phlegm',
        'mucus': 'phlegm',
        'bloody stool': 'bloody_stool',
        'blood in stool': 'bloody_stool',
        'anal pain': 'pain_in_anal_region',
        'rectal pain': 'pain_in_anal_region',
        'painful bowel': 'pain_during_bowel_movements',
        'bowel pain': 'pain_during_bowel_movements',
        'gas': 'passage_of_gases',
        'bloating': 'distention_of_abdomen',
        'swollen stomach': 'swelling_of_stomach',
        'stomach swelling': 'swelling_of_stomach',
        'swollen lymph nodes': 'swelled_lymph_nodes',
        'swollen glands': 'swelled_lymph_nodes',
        'bruising': 'bruising',
        'bruises': 'bruising',
        'cramps': 'cramps',
        'muscle cramps': 'cramps',
        'obesity': 'obesity',
        'overweight': 'obesity',
        'thyroid problems': 'enlarged_thyroid',
        'thyroid swelling': 'enlarged_thyroid',
        'brittle nails': 'brittle_nails',
        'weak nails': 'brittle_nails',
        'swollen extremities': 'swollen_extremeties',
        'swollen hands': 'swollen_extremeties',
        'swollen feet': 'swollen_extremeties',
        'puffy face': 'puffy_face_and_eyes',
        'swollen face': 'puffy_face_and_eyes',
        'slurred speech': 'slurred_speech',
        'speech problems': 'slurred_speech',
        'loss of balance': 'loss_of_balance',
        'balance problems': 'loss_of_balance',
        'unsteady': 'unsteadiness',
        'unsteady gait': 'unsteadiness',
        'weakness one side': 'weakness_of_one_body_side',
        'one sided weakness': 'weakness_of_one_body_side',
        'loss of smell': 'loss_of_smell',
        'no smell': 'loss_of_smell',
        'bladder problems': 'bladder_discomfort',
        'bladder pain': 'bladder_discomfort',
        'foul urine': 'foul_smell_of_urine',
        'smelly urine': 'foul_smell_of_urine',
        'frequent urination': 'continuous_feel_of_urine',
        'urge to urinate': 'continuous_feel_of_urine',
        'internal itching': 'internal_itching',
        'toxic look': 'toxic_look_(typhos)',
        'sick appearance': 'toxic_look_(typhos)',
        'irritability': 'irritability',
        'irritable': 'irritability',
        'altered consciousness': 'altered_sensorium',
        'confusion': 'altered_sensorium',
        'red spots': 'red_spots_over_body',
        'body spots': 'red_spots_over_body',
        'irregular periods': 'abnormal_menstruation',
        'period problems': 'abnormal_menstruation',
        'skin patches': 'dischromic _patches',
        'discolored patches': 'dischromic _patches',
        'concentration problems': 'lack_of_concentration',
        'can\'t concentrate': 'lack_of_concentration',
        'coma': 'coma',
        'unconscious': 'coma',
        'stomach bleeding': 'stomach_bleeding',
        'blood in stomach': 'stomach_bleeding',
        'alcohol history': 'history_of_alcohol_consumption',
        'drinking history': 'history_of_alcohol_consumption',
        'blood in sputum': 'blood_in_sputum',
        'coughing blood': 'blood_in_sputum',
        'calf veins': 'prominent_veins_on_calf',
        'visible veins': 'prominent_veins_on_calf',
        'heart palpitations': 'palpitations',
        'racing heart': 'palpitations',
        'painful walking': 'painful_walking',
        'walking pain': 'painful_walking',
        'pimples': 'pus_filled_pimples',
        'acne': 'pus_filled_pimples',
        'blackheads': 'blackheads',
        'scarring': 'scurring',
        'scars': 'scurring',
        'skin peeling': 'skin_peeling',
        'peeling skin': 'skin_peeling',
        'silver dusting': 'silver_like_dusting',
        'nail dents': 'small_dents_in_nails',
        'nail problems': 'small_dents_in_nails',
        'inflamed nails': 'inflammatory_nails',
        'nail inflammation': 'inflammatory_nails',
        'blisters': 'blister',
        'blister': 'blister',
        'nose sore': 'red_sore_around_nose',
        'nose redness': 'red_sore_around_nose',
        'yellow crust': 'yellow_crust_ooze',
        'crusty discharge': 'yellow_crust_ooze',
        'spinning sensation': 'spinning_sensation',
        'difficulty swallowing': 'difficulty_swallowing',
        'swollen tonsils': 'swollen_tonsils',
        'rapid pulse': 'rapid_pulse',
        'dry mouth': 'dry_mouth',
        'night vision problems': 'night_vision_problems',
        'faded colors': 'faded_colors' ,
        'vision loss' : 'vision_loss' ,
        'facial pain': 'facial_pain',
        'frequent urination': 'frequent_urination',
        'acid taste': 'acid_taste',
        'nasal congestion': 'nasal_congestion',
        'bad breath': 'bad_breath',
        'blood in urine': 'blood_in_urine',
        'chronic cough': 'chronic_cough',
        'dry skin': 'dry_skin',
        'high temperature': 'high_body_temperature',
        'heatburn': 'heatburn',
        'sore throat': 'sore_throat',
        'leg pain': 'leg_pain',
        'shoulder pain': 'shoulder_pain',
        'lower back pain': 'lower_back_pain',
        'chest discomfort': 'chest_discomfort',
    }
    
    matched_symptoms = []
    for symptom in patient_symptoms:
        symptom_lower = symptom.lower().strip()
        
        #first try exact match
        if symptom_lower in symptoms_dict:
            matched_symptoms.append(symptom_lower)
        #then try mapping
        elif symptom_lower in symptom_mapping:
            mapped_symptom = symptom_mapping[symptom_lower]
            if mapped_symptom in symptoms_dict:
                matched_symptoms.append(mapped_symptom)
        #try partial matching
        else:
            for key in symptoms_dict.keys():
                if symptom_lower in key or key in symptom_lower:
                    matched_symptoms.append(key)
                    break
    
    if not matched_symptoms:
        raise ValueError(f"No matching symptoms found for: {', '.join(patient_symptoms)}")
    
    for item in matched_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    
    return svc.predict([input_vector])[0]


#creating routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' , methods=['POST','GET'])
def predict():
    if request.method=='POST':
        try:
            symptoms = request.form.get('symptoms')
            if not symptoms or symptoms.strip() == '':
                return render_template('index.html', message='Please enter at least one symptom.')
            
            user_symptoms = [s.strip() for s in symptoms.split(",")]
            user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
            
            #filter out empty strings
            user_symptoms = [s for s in user_symptoms if s.strip()]
            
            if not user_symptoms:
                return render_template('index.html', message='Please enter valid symptoms.')
            
            predicted_disease = get_predicted_value(user_symptoms)
            descr, pre, med, diet, wrkout = helper(predicted_disease)

            my_precautions = []
            if pre and len(pre) > 0:
                for i in pre[0]:
                    if pd.isna(i):
                        continue
                    i_str = str(i).strip()
                    if i_str:
                        my_precautions.append(i_str)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=descr,
                                   my_precautions=my_precautions, medications=med, my_diet=diet,
                                   workout=wrkout, symptoms_input=symptoms)
        
        except ValueError as e:
            return render_template('index.html', message=f'Error: {str(e)}. Please try different symptoms or check spelling.')
        except Exception as e:
            return render_template('index.html', message=f'An error occurred: {str(e)}. Please try again.')





@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')



#python main
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


# House Price Predictor

This project is a Flask-based web application that predicts house prices using a trained machine learning model. Users can input house features via a web form or send JSON data through an API to receive price predictions. The ML model is trained using a dataset of housing features and prices.
- The **Flask web application** (`app.py`) and **HTML frontend** (`index.html`) were scaffolded and refined using AI-based tools.
- The **machine learning model training** (`house.py`) and data preprocessing logic were **written manually by the author** without AI automation.

This hybrid approach allowed rapid deployment of the web interface while maintaining full control and customization over the ML pipeline.


---


## Dataset

This project uses the **[Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)** from Kaggle, published by [Yasser H](https://www.kaggle.com/yasserh).

- Features include:
  - Area, bedrooms, bathrooms, stories
  - Main road, guest room, basement
  - Air conditioning, parking
- Target variable: **Price**

This dataset was saved as `Housing.csv` and used in `house.py` to train a `RandomForestRegressor`.

---


## Model Training

The model is trained using `house.py` with the dataset above. LabelEncoders are used for:
- `mainroad`
- `guestroom`
- `basement`
- `airconditioning`

The model and encoders are serialized using Pickle and loaded by the Flask app for prediction.


---


## Features

- Predict house prices instantly
- Built with Flask for web and API integration
- Trained using Random Forest Regressor
- Encodes categorical features using LabelEncoders
- Accepts both HTML form input and JSON API requests

---


##  Technologies Used

- Python 3
- Flask (Backend Web Framework)
- Scikit-learn (ML Model & Encoders)
- Pandas & NumPy (Data Processing)
- HTML + JavaScript (Frontend)
- Pickle (Model & Encoder Storage)
- Matplotlib/Seaborn (for EDA in development)

---


##  Project Structure

house-price-predictor/

├── app.py `Flask app (web + API)`

├── house.py `ML training script`

├── Housing.csv `Dataset used to train the model`

├── house_price_model.pkl `Trained RandomForest model`

├── main_road.pkl `LabelEncoder for 'mainroad`

├── guest_room.pkl `LabelEncoder for 'guestroom`

├── basement.pkl `LabelEncoder for 'basement`

├── air_conditioning.pkl `LabelEncoder for airconditioning`

├── requirements.txt `Python dependencies`

└── templates/

  └── index.html `Web frontend form`


---


## Project Workflow

Step 1: Load Dataset

  └── Housing.csv
  

Step 2: Train Model (house.py)

  ├── Encode categorical features (LabelEncoder)
  
  ├── Train RandomForestRegressor
  
  └── Save:
  
      ├── house_price_model.pkl
      
      ├── main_road.pkl
      
      ├── guest_room.pkl
      
      ├── basement.pkl
      
      └── air_conditioning.pkl
      

Step 3: Run Flask App (app.py)

  ├── Load model and encoders
  
  ├── Serve HTML form via /
  
  └── Expose API endpoint via /api/predict
  

Step 4: User Interaction

  ├── User inputs form OR sends JSON
  
  └── Flask processes → model predicts → returns predicted price
  

Step 5: Output

  ├── HTML view displays price
  
  └── OR API responds with JSON { "prediction": value }
  


---


##  How to Run the Project Locally

1. **Download the Project Files:**
   - Make sure you have all the following files and folders:
     - `app.py`
     - `house.py`
     - `Housing.csv`
     - All `.pkl` files (model and encoders)
     - A folder named `templates` containing `index.html`
     - `requirements.txt`
     - `README.md`

2. **Install Python:**
   - Ensure that Python 3 is installed on your system.
   - You can verify this by opening any terminal or command prompt and typing `python --version`.

3. **Install Required Libraries:**
   - Open your terminal or command prompt.
   - Install the required libraries listed in `requirements.txt` manually using pip or a package manager.
   - Libraries include:
     - Flask
     - Flask-Cors
     - numpy
     - pandas
     - scikit-learn
     - matplotlib
     - seaborn

4. **Train the Model (Optional):**
   - Open and run `house.py` if you want to retrain the model using `Housing.csv`.
   - This will generate the required model and encoder files (`.pkl`).

5. **Start the Web Application:**
   - Run the file `app.py` using Python.
   - It will start a local development server.

6. **Open the Web Page:**
   - Open any browser and go to `http://127.0.0.1:5000/`.
   - You will see the House Price Prediction form.

7. **Use the Application:**
   - Fill in the form with house details like area, number of bedrooms, availability of guest room, etc.
   - Click **Predict** to get the estimated house price.



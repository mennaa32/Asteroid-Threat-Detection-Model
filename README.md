# 🚀 Asteroid Threat Detection Model

This project uses real-time data from NASA's Near-Earth Object (NEO) API to detect whether an asteroid is potentially hazardous to Earth using machine learning.

---

## 🧠 Overview

An intelligent AI system that:
- Fetches asteroid data from NASA API
- Cleans and analyzes the data
- Trains a Random Forest model to classify threats


---

## 🌐 Data Source

- NASA NEO API  
- 

---

## 🔎 Workflow

1. **Data Collection**  
   Retrieves asteroid data for the last 5 days from NASA API.

2. **Data Cleaning & Preprocessing**  
   - Removes duplicates/missing values
   - Converts columns to proper data types
   - Scales numeric features

3. **Exploratory Data Analysis (EDA)**  
   - Visualizes asteroid velocity, miss distance, and danger distribution.

4. **Model Training**  
   - Random Forest Classifier
   - SMOTE used for handling imbalanced classes


## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 Future Ideas

- Deploy as a web app
- Add real-time notification
- Historical pattern analysis

---

## 👩‍💻 Developed by

**Menna Sayed**  
Faculty of Computers and Information – Class of 2025  
Supervised by: Egyptian Space Agency

# 🏡 House Price Predictor

A **Machine Learning + Streamlit-based web application** that predicts house prices based on Indian real estate data such as bedrooms, bathrooms, property size, furnishing status, property age, and location tier.  
Built to help buyers, sellers, and real estate agents make **data-driven decisions**.

---

## 💡 Problem Statement
The real estate market in India can be unpredictable due to factors like location, property age, size, and furnishing. Buyers often overpay or miss good deals because they lack quick, data-backed insights.

---

## 💡 Our Solution
We created **House Price Predictor**, a simple yet powerful web app that:
- Predicts property prices using a trained ML model.
- Adjusts predictions based on **location tier**, **property furnishing**, and **age**.
- Visualizes price trends to help users make informed choices.

---

## 🚀 Features
- 💡 **Price Prediction** – Estimate property prices instantly.
- 📈 **Detailed Breakdown** – Price per sq. ft and total value.
- 📊 **Visual Insights** – Price distribution & size vs price graphs.
- 🏙️ **Location Tier Support** – Metro, Tier 2, Tier 3.
- 🛋️ **Furnishing & Age Adjustments** – Fine-tune predictions.
- 🎨 **Clean UI** – Indian-themed, responsive design.
- 💾 **Dataset-backed Accuracy** – Trained on real Indian housing data.

---

## 📊 Dataset Details
The model is trained on a dataset with the following columns:

| Column Name       | Description |
|-------------------|-------------|
| **beds**          | Number of bedrooms 🛏️ |
| **baths**         | Number of bathrooms 🚿 |
| **size**          | Built-up area of the property |
| **size_units**    | Units of size (e.g., sq. ft, sq. m) |
| **lot_size**      | Land area size |
| **lot_size_units**| Units for lot size (e.g., sq. ft, acres) |
| **zip_code**      | Postal code of the property |
| **price**         | Final property price in ₹ |

---

## 🛠 Tech Stack
- **Frontend & UI:** Streamlit  
- **Backend & ML:** Python (Scikit-learn, Pandas, NumPy)  
- **Data Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit Cloud / Render  

---
---

## 📸 Screenshots
| Home Page | Prediction Form |
|-----------|-----------------|
| ![Home](1.jpg) | ![Form](2.jpg) |

| | |
|-------------------|-------------|
| ![Output](3.jpg) | ![Graph](4.jpg) |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.7+
- pip installed
- Streamlit

# Clone the repository
git clone https://github.com/your-username/House-Price-Predictor.git

# Navigate to the project folder
cd House-Price-Predictor

# Install dependencies
pip install -r requirements.txt

# To Run
streamlit run house.py

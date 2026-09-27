# Delivery Time Predictor 🛵

A Streamlit web app that predicts food delivery time (in minutes) using a trained XGBoost regression model.

## Features

- Interactive form to enter delivery details (rider age & rating, traffic, weather, distance, vehicle type, etc.)
- Instant delivery time prediction on click
- View of the exact feature vector sent to the model, for debugging/transparency

## Demo

![App screenshot placeholder]model/images/frontend.png)

## Tech Stack

- **Frontend/App**: [Streamlit](https://streamlit.io/)
- **Model**: XGBoost Regressor (`xgb_best.pkl`)
- **Preprocessing**: scikit-learn `StandardScaler` (`standard_scaler.pkl`)

## Machine Learning Pipeline

The model was trained on food delivery data (similar to the Zomato Delivery Time dataset) to predict delivery time in minutes based on 12 raw input fields.

**Raw inputs:**
| Field | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery rider |
| `Delivery_person_Ratings` | Rider's average rating |
| `Weatherconditions` | Weather at time of delivery (Sunny, Stormy, Fog, etc.) |
| `Road_traffic_density` | Traffic level (Low, Medium, High, Jam) |
| `Vehicle_condition` | Condition rating of the delivery vehicle (0–3) |
| `Type_of_order` | Buffet, Drinks, Meal, or Snack |
| `Type_of_vehicle` | Bicycle, motorcycle, scooter, or electric scooter |
| `multiple_deliveries` | Number of other deliveries bundled in the same trip |
| `Festival` | Whether it's a festival day (Yes/No) |
| `City` | City type (Metropolitan, Urban, Semi-Urban) |
| `distance_km` | Distance between restaurant and drop-off location |
| `order_hour` | Hour of day the order was placed (0–23) |

**Feature engineering:**
- `Road_traffic_density` is ordinal-encoded (`Low=0, Medium=1, High=2, Jam=3`)
- `Type_of_order`, `Type_of_vehicle`, `Festival`, `City`, and `Weatherconditions` are one-hot encoded (drop-first), producing 21 total model features
- All 21 features are scaled with a fitted `StandardScaler` before being passed to the model

**Model:** XGBoost Regressor (`n_estimators=500`, `max_depth=8`, `learning_rate=0.01`), trained to minimize error on delivery time (minutes).

## Exploratory Data Analysis (EDA)

<!-- TODO: Add a short summary of your EDA findings here, e.g. -->
<!-- - Distribution of delivery times -->
<!-- - Correlation between distance and delivery time -->
<!-- - Impact of traffic/weather/festival on delivery time -->
<!-- - Any outliers or missing values handled -->

Key observations from the exploratory data analysis:


- **Data health check** — missing values, duplicates, dtype validation
- **Target distribution** — histogram of `Time_taken` with outlier inspection
- **Distance vs. delivery time** — scatter plot to validate the primary predictive signal
- **Traffic density vs. delivery time** — boxplot across Low/Medium/High/Jam
- **Weather conditions vs. delivery time** — boxplot across weather categories
- **Correlation heatmap** — numeric features vs. target
- **Outlier handling** — rows with invalid/extreme values (e.g. zero distance, unrealistic delivery times) were removed/capped; see `FOOD_TIME_DEL_PRID_MOD.ipynb` for details


## Model Performance

Comparison of models tried during experimentation (final model used in the app: **XGBoost**):

| Metric | Linear Regression | Random Forest | XGBoost |
|---|---|---|---|
| MAE (Mean Absolute Error)       | 4.9301 |3.1935 |3.1756 |
| RMSE (Root Mean Squared Error)  |6.2125 | 4.0286 |3.9102|
| R² Score                        |0.5693 | 0.8189 | 0.8294
 |

## Project Structure

```
delivery-time-predictor/
├── app.py                  # Streamlit app
├── xgb_best.pkl             # Trained XGBoost model
├── standard_scaler.pkl      # Fitted StandardScaler
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/delivery-time-predictor.git
cd delivery-time-predictor
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Notes / Known Assumptions

- `Road_traffic_density` is assumed to be encoded as `Low=0, Medium=1, High=2, Jam=3` based on the saved scaler's feature order. If your training pipeline used a different mapping, update `TRAFFIC_MAP` in `app.py`.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

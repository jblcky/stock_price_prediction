# 📈 CSPX Stock Price Prediction with LSTM

This project demonstrates a **CPU-friendly LSTM model** for predicting the next-day closing price of the CSPX (S&P 500 ETF) using historical market data. The project includes a simple **Streamlit front end** for interactive predictions.

---

## 🚀 Steps to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/stock_price_prediction.git
cd stock_price_prediction


### Step 2: Install Dependencies
Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt


### Step 3: Train or Load the Model
You can either:

- **Train the model** using the provided Jupyter notebook (`notebooks/stock_lstm.ipynb`), or
- **Use the pre-trained model files** included in this repo:
  - `lstm_model.h5` (LSTM model)
  - `feature_scaler.pkl` (feature scaler)
  - `target_scaler.pkl` (target scaler)


### Step 4: Run the Streamlit App
Launch the Streamlit front end locally:

```bash
streamlit run app.py


### Step 5: Upload Data & Predict
- Upload a CSPX historical CSV file through the Streamlit interface.
- The app will display the last few rows of data.
- Get an instant prediction of the **next-day closing price**.

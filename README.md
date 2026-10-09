# E-Commerce Sales & Customer Analytics

A beginner-friendly Flask analytics dashboard for:
- Product performance
- Pricing patterns
- Customer ratings and reviews
- Basic sentiment analysis
- Demand/category trends
- Market opportunity identification
- Optional historical sales forecasting
- User registration, login, profile and logout
- CSV dataset upload

## 1. Install Python
Use Python 3.11 or newer.

## 2. Open terminal in this project folder

### Windows
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run
```powershell
python app.py
```

Open:
http://127.0.0.1:5000
## 4. Live Demo 
https://ecommerce-sales-customer-analytics-gwsi.onrender.com

## 5. Demo account
Register your own account from the Register page.

Example:
- Name: Geetha
- Email: geetha@example.com
- Password: geetha123

## 6. Dataset
Put your CSV file at:
`data/ecommerce_data.csv`

The application accepts common column names. Recommended columns:

| Column | Example |
|---|---|
| product_name | Wireless Headphones |
| category | Electronics |
| price | 2499 |
| rating | 4.5 |
| review | Excellent sound quality |
| customer_id | C101 |
| date | 2026-01-15 |
| quantity | 3 |
| sales | 7497 |

If `sales` is not present but `price` and `quantity` are present, sales is calculated automatically.

## 7. Login/Profile/Logout
- `/register` creates an account.
- `/login` authenticates the account.
- `/dashboard` is protected and requires login.
- `/profile` shows the logged-in user's profile.
- `/logout` clears the session.

Passwords are stored as secure hashes, not plain text.

## 7. Important project limitation
This project analyzes e-commerce data. It does not guarantee stock-market returns or investment decisions. "Market opportunities" means business/product opportunities inferred from the supplied e-commerce data.

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import os
import re

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "change-this-secret-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ecommerce.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

UPLOAD_FOLDER = os.path.join(
    app.root_path, "data", "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# 2. USER DATABASE
# --------------------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(
        db.String(150), unique=True, nullable=False
    )

    password = db.Column(db.String(255), nullable=False)

    created_at = db.Column(
        db.DateTime, default=datetime.utcnow
    )


# --------------------------------------------------
# 3. LOGIN PROTECTION
# --------------------------------------------------

def login_required(function):
    @wraps(function)
    def decorated(*args, **kwargs):

        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated


# --------------------------------------------------
# 4. GET THE CURRENT USER'S DATASET PATH
# --------------------------------------------------

def get_user_data_path(user_id=None):

    if user_id is None:
        user_id = session.get("user_id")

    if user_id is None:
        return None

    return os.path.join(
        UPLOAD_FOLDER, f"user_{user_id}.csv"
    )


# --------------------------------------------------
# 5. LOAD AND CLEAN CSV DATA
# --------------------------------------------------

def load_data(file_path):

    if not file_path or not os.path.isfile(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError):
        return pd.DataFrame()

    if df.empty or len(df.columns) == 0:
        return pd.DataFrame()

    # Standardize column names
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    # Recognize alternative column names
    aliases = {
        "product_name": [
            "product", "product_title", "name"
        ],
        "category": [
            "product_category", "category_name"
        ],
        "price": [
            "selling_price", "discounted_price", "amount"
        ],
        "rating": [
            "stars", "review_rating"
        ],
        "review": [
            "review_text", "reviews", "comment"
        ],
        "customer_id": [
            "buyer_id", "customer", "user_id"
        ],
        "date": [
            "order_date", "review_date", "timestamp"
        ],
        "quantity": [
            "qty", "units_sold", "sales_quantity"
        ],
        "sales": [
            "revenue", "total_sales", "sales_amount"
        ]
    }

    for target, alternatives in aliases.items():

        if target not in df.columns:
            for column in alternatives:
                if column in df.columns:
                    df[target] = df[column]
                    break

    # Convert numeric columns safely
    for column in ["price", "rating", "quantity", "sales"]:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("£", "", regex=False),
                errors="coerce"
            )

    # Calculate sales only when the required fields exist
    if "sales" not in df.columns:

        if "price" in df.columns and "quantity" in df.columns:
            df["sales"] = (
                df["price"].fillna(0)
                * df["quantity"].fillna(0)
            )
        else:
            df["sales"] = 0.0

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    return df


# --------------------------------------------------
# 6. CUSTOMER SENTIMENT ANALYSIS
# --------------------------------------------------

def sentiment_label(text):

    if not isinstance(text, str) or not text.strip():
        return "Neutral"

    positive_words = {
        "good", "great", "excellent", "amazing",
        "love", "best", "happy", "perfect",
        "fast", "useful", "worth", "wonderful",
        "comfortable", "quality", "recommend"
    }

    negative_words = {
        "bad", "poor", "worst", "hate", "slow",
        "broken", "terrible", "disappointed",
        "expensive", "waste", "defective",
        "uncomfortable", "damaged"
    }

    words = re.findall(r"[a-zA-Z]+", text.lower())

    positive_count = sum(
        word in positive_words for word in words
    )

    negative_count = sum(
        word in negative_words for word in words
    )

    if positive_count > negative_count:
        return "Positive"

    if negative_count > positive_count:
        return "Negative"

    return "Neutral"


# --------------------------------------------------
# 7. ANALYZE THE CURRENT USER'S DATASET
# --------------------------------------------------

def analytics(user_id=None):

    file_path = get_user_data_path(user_id)
    df = load_data(file_path)

    # No uploaded dataset means no analytics
    if df.empty:
        return {"empty": True}

    total_products = (
        int(df["product_name"].nunique())
        if "product_name" in df.columns else len(df)
    )

    total_categories = (
        int(df["category"].nunique())
        if "category" in df.columns else 0
    )

    total_reviews = (
        int(df["review"].notna().sum())
        if "review" in df.columns else 0
    )

    avg_rating = (
        round(float(df["rating"].mean()), 2)
        if "rating" in df.columns and df["rating"].notna().any()
        else 0
    )

    total_sales = round(
        float(df["sales"].fillna(0).sum()), 2
    )

    # Sales by category
    top_categories = []

    if "category" in df.columns:

        if df["sales"].sum() > 0:
            category_data = (
                df.groupby("category")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
        else:
            category_data = (
                df["category"].value_counts().head(10)
            )

        top_categories = [
            {
                "name": str(name),
                "value": round(float(value), 2)
            }
            for name, value in category_data.items()
        ]

    # Top products
    top_products = []

    if "product_name" in df.columns:

        if df["sales"].sum() > 0:
            product_data = (
                df.groupby("product_name")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
        else:
            product_data = (
                df["product_name"].value_counts().head(10)
            )

        top_products = [
            {
                "name": str(name),
                "value": round(float(value), 2)
            }
            for name, value in product_data.items()
        ]

    # Rating distribution
    ratings = {}

    if "rating" in df.columns:

        rating_data = (
            df["rating"].dropna().round()
            .value_counts().sort_index()
        )

        ratings = {
            str(int(rating)): int(count)
            for rating, count in rating_data.items()
        }

    # Customer sentiment
    sentiment = {}

    if "review" in df.columns:

        sentiment_data = (
            df["review"].dropna()
            .apply(sentiment_label)
            .value_counts()
        )

        sentiment = {
            str(label): int(count)
            for label, count in sentiment_data.items()
        }

    # Average price by category
    price_by_category = []

    if "category" in df.columns and "price" in df.columns:

        price_data = (
            df.dropna(subset=["price"])
            .groupby("category")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        price_by_category = [
            {
                "name": str(name),
                "value": round(float(value), 2)
            }
            for name, value in price_data.items()
        ]

    # Potential market opportunities
    opportunities = []

    if (
        "product_name" in df.columns
        and "rating" in df.columns
        and df["rating"].notna().any()
    ):

        grouped = df.groupby("product_name").agg(
            sales=("sales", "sum"),
            avg_rating=("rating", "mean")
        ).reset_index()

        median_sales = grouped["sales"].median()

        candidates = grouped[
            (grouped["sales"] >= median_sales)
            & (grouped["avg_rating"] >= 4)
        ].sort_values("sales", ascending=False)

        opportunities = [
            {
                "product": str(row["product_name"]),
                "reason": (
                    "Strong rating and sales performance"
                )
            }
            for _, row in candidates.head(10).iterrows()
        ]

    # Monthly sales forecasting
    forecast = []

    if (
        "date" in df.columns
        and df["date"].notna().any()
        and df["sales"].notna().any()
    ):

        dated = df.dropna(subset=["date"]).copy()

        dated["month"] = (
            dated["date"].dt.to_period("M").astype(str)
        )

        monthly = (
            dated.groupby("month")["sales"]
            .sum()
            .reset_index()
            .sort_values("month")
        )

        if len(monthly) >= 4:

            monthly["period_num"] = np.arange(len(monthly))

            model = LinearRegression()

            model.fit(
                monthly[["period_num"]],
                monthly["sales"]
            )

            future_x = np.arange(
                len(monthly), len(monthly) + 3
            ).reshape(-1, 1)

            predictions = model.predict(future_x)

            last_month = pd.Period(
                monthly["month"].iloc[-1], freq="M"
            )

            for index, prediction in enumerate(
                predictions, start=1
            ):

                future_month = last_month + index

                forecast.append({
                    "period": str(future_month),
                    "value": round(
                        max(0.0, float(prediction)), 2
                    )
                })

    return {
        "empty": False,
        "total_products": total_products,
        "total_categories": total_categories,
        "total_reviews": total_reviews,
        "avg_rating": avg_rating,
        "total_sales": total_sales,
        "top_categories": top_categories,
        "top_products": top_products,
        "ratings": ratings,
        "sentiment": sentiment,
        "price_by_category": price_by_category,
        "opportunities": opportunities,
        "forecast": forecast
    }


# --------------------------------------------------
# 8. HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# 9. REGISTER
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 6:
            flash(
                "Enter all details. Password must contain "
                "at least 6 characters.",
                "danger"
            )
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash(
                "Email already registered. Please log in.",
                "warning"
            )
            return redirect(url_for("login"))

        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful. Please log in.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# --------------------------------------------------
# 10. LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session.clear()
            session["user_id"] = user.id
            session["user_name"] = user.name

            flash("Login successful.", "success")

            # If the user has no CSV, the dashboard shows
            # the upload prompt instead of analytics.
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


# --------------------------------------------------
# 11. LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.", "success")

    return redirect(url_for("index"))


# --------------------------------------------------
# 12. DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    user_data = analytics(session["user_id"])

    return render_template(
        "dashboard.html",
        data=user_data
    )


# --------------------------------------------------
# 13. PROFILE
# --------------------------------------------------

@app.route("/profile")
@login_required
def profile():

    user = db.session.get(
        User, session["user_id"]
    )

    return render_template(
        "profile.html",
        user=user
    )


# --------------------------------------------------
# 14. UPLOAD CSV
# --------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():

    if request.method == "POST":

        file = request.files.get("file")

        if not file or not file.filename:
            flash("Please select a CSV file.", "danger")
            return redirect(url_for("upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("upload"))

        # Validate the CSV before saving it
        try:
            df_check = pd.read_csv(file, nrows=5)

            if df_check.empty or len(df_check.columns) == 0:
                flash(
                    "The CSV is empty or has no columns.",
                    "danger"
                )
                return redirect(url_for("upload"))

            # Reset the uploaded file pointer
            file.stream.seek(0)

        except Exception:
            flash(
                "Unable to read this CSV. Please check the file.",
                "danger"
            )
            return redirect(url_for("upload"))

        # Each user gets a separate CSV file
        user_path = get_user_data_path(
            session["user_id"]
        )

        os.makedirs(
            os.path.dirname(user_path),
            exist_ok=True
        )

        try:
            file.save(user_path)

            # Confirm that the saved CSV can be read
            saved_df = load_data(user_path)

            if saved_df.empty:
                os.remove(user_path)

                flash(
                    "The uploaded CSV could not be analyzed.",
                    "danger"
                )
                return redirect(url_for("upload"))

        except Exception:
            flash(
                "An error occurred while saving the CSV.",
                "danger"
            )
            return redirect(url_for("upload"))

        flash(
            "Dataset uploaded and analyzed successfully.",
            "success"
        )

        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# --------------------------------------------------
# 15. COMMON TEMPLATE VARIABLES
# --------------------------------------------------

@app.context_processor
def inject_user():

    return {
        "logged_in": "user_id" in session,
        "user_name": session.get("user_name")
    }


# --------------------------------------------------
# 16. CREATE DATABASE AND START SERVER
# --------------------------------------------------

with app.app_context():
    db.create_all()


if __name__ == "__main__":
  app.run(debug=True)

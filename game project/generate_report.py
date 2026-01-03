import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from src.preprocessing import clean_data
from src.model import train_model

DATA_PATH = "data/all_games.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")
df = clean_data(df)

# Hit/Flop
df['hit_flop'] = df['meta_score'].apply(lambda x: 'Hit' if x >= 75 else 'Flop')

# Train model
model = train_model(df)

# Platform vs Meta Score
plt.figure(figsize=(8,5))
platform_score = df.groupby("platform")["meta_score"].mean().sort_values(ascending=False).reset_index()
plt.bar(platform_score['platform'], platform_score['meta_score'], color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("Average Meta Score")
plt.title("Average Meta Score by Platform")
plt.tight_layout()
plt.savefig("platform_score.png")
plt.close()

# Hit/Flop pie chart
plt.figure(figsize=(6,6))
hit_flop_counts = df['hit_flop'].value_counts()
plt.pie(hit_flop_counts, labels=hit_flop_counts.index, autopct='%1.1f%%', colors=['green','red'])
plt.title("Hit vs Flop Games")
plt.savefig("hit_flop.png")
plt.close()

# PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.multi_cell(0, 10, "Predicting Game Success Using Sales & Reviews Data", align="C")
pdf.ln(10)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, "College Project Report\nBy: CHINRAJ\nDate: 2026-01-01", align="C")

# Dataset info
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Dataset Information", ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nColumn Names: {df.columns.tolist()}")

# Model performance
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0,10, "Model Performance (Random Forest)", ln=True)
pdf.set_font("Arial", '', 12)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
features = df.select_dtypes(include=["int64","float64"]).drop(columns=['meta_score'])
X_train, X_test, y_train, y_test = train_test_split(features, df['meta_score'], test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
pdf.multi_cell(0,8, f"MAE: {mean_absolute_error(y_test, y_pred):.4f}\nR² Score: {r2_score(y_test, y_pred):.4f}")

# EDA charts
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "EDA Graphs", ln=True)
pdf.image("platform_score.png", w=180)
pdf.ln(10)
pdf.image("hit_flop.png", w=120)

pdf.output("Game_Success_Report.pdf")
print("✅ PDF report generated successfully")

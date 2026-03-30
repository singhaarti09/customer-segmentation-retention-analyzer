import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Save model

# ========================= LOAD & CLEAN =========================
df = pd.read_csv("dataset.csv", sep="\t")
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.dropna()

df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
df["Churn"] = df["Churn"].map({"true": 1, "false": 0, "yes": 1, "no": 0})
df = df.dropna(subset=["Churn"])

df["International_plan"] = df["International_plan"].map({"Yes": 1, "No": 0})
df["Voice_mail_plan"] = df["Voice_mail_plan"].map({"Yes": 1, "No": 0})

le = LabelEncoder()
df["State"] = le.fit_transform(df["State"])

# ========================= TRAIN MODEL =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# ========================= MODEL EVALUATION (UPDATED) =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Model Accuracy: {:.2f}%".format(accuracy * 100))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# ========================= SEGMENTATION =========================
features_cluster = [
    'Total_day_minutes', 'Total_day_charge',
    'Total_eve_minutes', 'Total_eve_charge',
    'Total_night_minutes', 'Total_night_charge',
    'Customer_service_calls', 'Account_length'
]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(df[features_cluster])

# ========================= PREDICTIONS =========================
df["Churn_Prediction"] = model.predict(X)
df["Churn_Probability"] = model.predict_proba(X)[:, 1]  # Risk score

print("\n🎯 Segments:", df["Segment"].value_counts().to_dict())
print("🔄 Predicted Churn:", df["Churn_Prediction"].value_counts().to_dict())

# ========================= SAVE =========================
joblib.dump(model, 'churn_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

df.to_csv("output.csv", index=False)

print("\n✅ Model + Segments saved to output.csv!")
from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def home():
    try:
        data = pd.read_csv("output.csv")

        # ========================= TABLE =========================
        table = data.head(100).to_html(classes='table table-striped', index=False)

        # ========================= SEGMENT BAR CHART =========================
        plt.figure(figsize=(8,5))
        seg_spend = data.groupby('Segment')['Total_day_charge'].mean().round(2)
        seg_spend.plot(kind='bar')
        plt.title("Average Day Charges by Segment")
        plt.xlabel("Segment")
        plt.ylabel("Avg Charge")

        img1 = io.BytesIO()
        plt.savefig(img1, format='png', bbox_inches='tight')
        plt.close()
        img1.seek(0)
        chart_bar = base64.b64encode(img1.getvalue()).decode()

        # ========================= CHURN PIE CHART =========================
        plt.figure(figsize=(6,6))
        data["Churn_Prediction"].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%'
        )
        plt.title("Churn Distribution")

        img2 = io.BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        plt.close()
        img2.seek(0)
        chart_pie = base64.b64encode(img2.getvalue()).decode()

        # ========================= HIGH VALUE CUSTOMERS =========================
        high_value = data[data['Segment'] == 2].head(20)
        high_table = high_value.to_html(classes='table table-striped', index=False)

        # ========================= HIGH CHURN RISK =========================
        churn_risk = data.nlargest(20, 'Churn_Probability')
        churn_table = churn_risk.to_html(classes='table table-striped', index=False)

        # ========================= STATS =========================
        total_customers = len(data)
        churn_rate = round((data['Churn_Prediction'].sum() / total_customers) * 100, 2)

        return render_template(
            "index.html",
            tables=table,
            high=high_table,
            churn=churn_table,
            bar_chart=chart_bar,
            pie_chart=chart_pie,
            stats={
                'total': total_customers,
                'churn_rate': churn_rate
            }
        )

    except Exception as e:
        return f"❌ Error: {str(e)}<br><br>👉 Run 'python model.py' first!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
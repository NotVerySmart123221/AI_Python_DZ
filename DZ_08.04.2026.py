import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

df = pd.read_csv("tips.csv")

def interactive_analysis(
    days: tuple = ("All",),
    gender: str = "All",
    bill_range: tuple = (0.0, 60.0),
    tip_range: tuple = (0.0, 15.0),
    only_smokers: bool = False,
):
    data = df.copy()

    # filter
    if "All" not in days:
        data = data[data["day"].isin(days)]

    if gender != "All":
        data = data[data["sex"] == gender]

    data = data[(data["total_bill"] >= bill_range[0]) & (data["total_bill"] <= bill_range[1])]
    data = data[(data["tip"] >= tip_range[0]) & (data["tip"] <= tip_range[1])]

    if only_smokers:
        data = data[data["smoker"] == "Yes"]

    display(data.head(10))

    # metrics
    avg_bill = data["total_bill"].mean()
    avg_tip = data["tip"].mean()
    avg_ratio = (data["tip"] / data["total_bill"]).mean()
    count_orders = len(data)

    print(f"средний total_bill: {avg_bill:.2f}")
    print(f"средний tip: {avg_tip:.2f}")
    print(f"среднее отношение tip / total_bill: {avg_ratio:.2f}")
    print(f"количество заказов: {count_orders}")

    if gender == "All":
        mal = len(data[data["sex"] == "Male"])
        fem = len(data[data["sex"] == "Female"])
        print(f"мужчин: {mal}, женщин: {fem}")

    # visual
    if not data.empty:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        data["total_bill"].plot(kind="hist", ax=axes[0, 0], title="total bill")
        data["tip"].plot(kind="hist", ax=axes[0, 1], title="tip distribution")
        
        data.boxplot(column="tip", by="sex", ax=axes[1, 0])
        data.boxplot(column="total_bill", by="day", ax=axes[1, 1])

        data.plot.scatter(x="total_bill", y="tip", ax=axes[0, 2], title="scatter bill and tip")

        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    print("\nInfo")
    data.info()
    
    print("\nDescribe")
    display(data.describe())

widgets.interact(
    interactive_analysis,
    days = widgets.SelectMultiple(options=["All", "Thur", "Fri", "Sat", "Sun"], value=("All",)),
    gender = widgets.Dropdown(options=["All", "Male", "Female"], value="All"),
    bill_range = widgets.FloatRangeSlider(
        min=float(df['total_bill'].min()), 
        max=float(df['total_bill'].max()), 
        value=(df['total_bill'].min(), df['total_bill'].max()), 
    ),
    tip_range = widgets.FloatRangeSlider(
        min=float(df['tip'].min()), 
        max=float(df['tip'].max()), 
        value=(df['tip'].min(), df['tip'].max()), 
    ),
    only_smokers = widgets.Checkbox(value=False),
)

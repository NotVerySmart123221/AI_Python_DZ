import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipywidgets import interact, widgets

df = pd.read_csv("job_salary_prediction_dataset.csv")

plot_type = widgets.Dropdown(
    options=["Line", "Bar", "Scatter"],
    value="Line",
    description="Plot:"
)

x_col = widgets.Dropdown(
    options=df.columns,
    description="X:"
)

y_col = widgets.Dropdown(
    options=df.select_dtypes(include="number").columns,
    description="Y:"
)

group_ckeck = widgets.Checkbox(
    value=True,
    description="Do Group"
)

agg_func = widgets.Dropdown(
    options=["mean", "sum", "min", "max", "count"],
    value="mean",
    description="Agg:"
)

sort_check = widgets.Checkbox(
    value=True,
    description="Sort"
)

save_btn = widgets.Button(
    description="Save Plot",
    icon="download"
)

sort_asc = widgets.Checkbox(
    value=True,
    description="ascending"
)

range_slider = widgets.IntRangeSlider(
    value=[0, 100],
    min=-100,
    max=1000,
    step=1,
    continuous_update=False
)

ui = widgets.VBox([
    widgets.HBox([plot_type, x_col, y_col, agg_func]),
    widgets.HBox([
        group_ckeck, 
        sort_check, 
        sort_asc, 
        range_slider, 
        save_btn
    ])
])

def update_range(change):
    new_y = change['new']
    if df[new_y].dtype in ['int64', 'float64']:
        c_min, c_max = int(df[new_y].min()), int(df[new_y].max())
        range_slider.min = min(c_min, range_slider.min)
        range_slider.max = max(c_max, range_slider.max)
        range_slider.min = c_min
        range_slider.max = c_max
        range_slider.value = [c_min, c_max]

y_col.observe(update_range, names='value')

def update(plot_type, x_col, y_col, group_ckeck, agg_func, sort_check, sort_asc, range_val, save=False):
    if df[y_col].dtype in ['int64', 'float64']:
        working_df = df[(df[y_col] >= range_val[0]) & (df[y_col] <= range_val[1])]
    else:
        working_df = df

    total_records = len(working_df)
    
    plt.figure(figsize=(10, 5))
    
    if group_ckeck:
        group = working_df.groupby(x_col)[y_col].agg(agg_func)
        if sort_check:
            group = group.sort_values(ascending=sort_asc)
        x_data = group.index.astype(str)
        y_data = group.values
    else:
        show_df = working_df.sample(min(1000, len(working_df)))
        if sort_check:
            show_df = show_df.sort_values(y_col, ascending=sort_asc)
        x_data = show_df[x_col]
        y_data = show_df[y_col]
    if plot_type == "Bar": plt.bar(x_data, y_data)
    elif plot_type == "Scatter": plt.scatter(x_data, y_data, alpha=0.5)
    else: plt.plot(x_data, y_data)

    plt.title(f"{y_col} by {x_col}\n(recs: {total_records})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) 
    if save: plt.savefig("plot.png") 
    plt.show()

def save_plot(b):
    update(
           plot_type.value, 
           x_col.value, 
           y_col.value, 
           group_ckeck.value, 
           agg_func.value, 
           sort_check.value,
           sort_asc.value,
           range_slider.value,
           save=True
    )
save_btn.on_click(save_plot)

out = widgets.interactive_output(
    update,
    {
        "plot_type": plot_type,
        "x_col": x_col,
        "y_col": y_col,
        "group_ckeck": group_ckeck,
        "agg_func": agg_func,
        "sort_check": sort_check,
        "sort_asc": sort_asc,
        "range_val": range_slider
    }
)

display(ui, out)

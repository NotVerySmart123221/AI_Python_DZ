from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display, Markdown

# task 1

range_1 = widgets.SelectionRangeSlider(
    options=range(11),
    index=(0, 5),
    description='range 1:',
    orientation='horizontal'
)

op_dropdown = widgets.Dropdown(
    options={'add (+)': '+', 'subtract (-)': '-', 'multiply (*)': '*', 'divide (/)': '/'},
    value='*',
    description='op:',
)

range_2 = widgets.SelectionRangeSlider(
    options=range(11),
    index=(0, 5),
    description='range 2:',
    orientation='horizontal'
)

def table(r1, op, r2):
    text = f"""
# results ({op})
| num 1 | num 2 | result |
| :--- | :--- | :--- |
"""
    for n1 in r1:
        for n2 in r2:
            res = eval(f"{n1} {op} {n2}")
            text += f"| {n1} | {n2} | {round(res, 2)} |\n"
    display(Markdown(text))
interact(table, r1=range_1, op=op_dropdown, r2=range_2)

# task 2

# banwords = ["adipiscing", "lorem", "ipsum", "sit"]
# text_ipt = widgets.Textarea(
#     value='Lorem ipsum eiusmod sit sed, consectetur adipiscing elit.',
#     layout={'height': '100px'}
# )

# def filter_text(orig_text):
#     prod_text = orig_text
    
#     for word in banwords:
#         prod_text = prod_text.replace(word, "ROSKOMNADZOR")
#     text = f"""
# {prod_text}
# ---
# """
#     display(Markdown(text))
# interact(filter_text, orig_text=text_ipt)

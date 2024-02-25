import panel as pn
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import warnings
import io
import os


warnings.filterwarnings("ignore")
pn.extension("plotly", "tabulator", comms="vscode")
load_dotenv()

file_name = "data.csv"
data = pd.read_csv(file_name).drop(columns=["id"])
plot_pane = pn.pane.Plotly(sizing_mode="stretch_width")
file_input = pn.widgets.FileInput()
text_input = pn.widgets.TextInput(
    name="Question",
    placeholder="Ask a question from the CSV",
    sizing_mode="scale_width",
)
ask_button = pn.widgets.Button(name="Ask", button_type="primary", height=60)
load_button = pn.widgets.Button(name="Load", button_type="primary")
plot_button = pn.widgets.Button(name="Plot", button_type="primary")
chat_box = pn.widgets.ChatBox(
    value=[], message_hue=220, ascending=True, allow_input=False
)


def load_page(data, file_name):
    target = data.columns[-1]
    yaxis = pn.widgets.Select(
        name="Y axis",
        options=list(data.columns),
        value=list(data._get_numeric_data().columns)[0],
        disabled_options=list(
            set(data.columns) - set(data._get_numeric_data().columns)
        ),
    )
    xaxis = pn.widgets.Select(
        name="X axis",
        options=list(data.columns),
        value=list(data._get_numeric_data().columns)[1],
        disabled_options=list(
            set(data.columns) - set(data._get_numeric_data().columns)
        ),
    )
    plot = px.scatter(
        data,
        x=list(data._get_numeric_data().columns)[0],
        y=list(data._get_numeric_data().columns)[1],
        color=target,
    )
    table = pn.widgets.Tabulator(data)
    agent = create_csv_agent(
        OpenAI(temperature=0), file_name, verbose=True, return_intermediate_steps=True
    )

    return target, yaxis, xaxis, plot, table, agent


target, yaxis, xaxis, plot_pane.object, table, agent = load_page(data, file_name)
plot_pane

template = pn.template.FastListTemplate(
    title="CSV Analyser",
    sidebar=[
        pn.Row(pn.pane.Markdown("# Analyze your CSVs")),
        pn.Row(pn.pane.Markdown("## Settings")),
        pn.Row(file_input, load_button),
        pn.Row(yaxis),
        pn.Row(xaxis),
        pn.Row(plot_button),
        pn.Row(plot_pane),
    ],
    main=[
        pn.Row(pn.Column(table), height=200),
        pn.Row(pn.Column(text_input), pn.Column(ask_button)),
        pn.Row(pn.Column(chat_box, sizing_mode="scale_width", scroll=True)),
    ],
    sidebar_width=420,
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)


def reloader(event):
    plot_pane = px.scatter(data, x=xaxis.value, y=yaxis.value, color=target)
    template.sidebar[6][0].object = plot_pane


plot_button.on_click(reloader)


def parse_file_input(event):

    global data, file_name

    value = file_input.value
    bytes_io = io.BytesIO(value)
    data = pd.read_csv(bytes_io)
    file_name = file_input.filename
    data.to_csv(file_name, index=False)

    global target, yaxis, xaxis, plot, table, agent

    target, yaxis, xaxis, plot, table, agent = load_page(data, file_name)

    template.main[0][0] = table
    template.sidebar[3][0] = yaxis
    template.sidebar[4][0] = xaxis
    template.sidebar[6][0].object = plot


load_button.on_click(parse_file_input)


def ask(event):
    query = text_input.value
    chat_box.append({"User": query})
    response = agent({"input": query})
    chat_box.append(
        {"Thought Process": [x[0].log for x in response["intermediate_steps"]]}
    )
    chat_box.append({"Bot": response["output"]})


ask_button.on_click(ask)


template.show(open=False, address="0.0.0.0", port=5006)

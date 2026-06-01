"""Planner tab — design RSA models / GLM contrasts from the project's stimuli.

Phase 1: renders the EmoC stimulus design (condition chips + partition table)
so the layout and data wiring are in place. Phase 5 will add interactive
model/contrast building that hands off to the Scheduler tab.
"""

import os
import sys

from dash import Dash, html, dcc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import stimuli, dash_kwargs

DARK_BG, PANEL_BG = "#ffffff", "#f3f5f9"
INK, MUTED, LINE = "#222222", "#667085", "#d5dbe5"

app = Dash(__name__, suppress_callback_exceptions=True, **dash_kwargs("PLANNER_URL_BASE"))
app.title = "Analysis Planner"


def _chip(species, label):
    color = stimuli.label_color(label)
    code = stimuli.condition_code(species, label)
    return html.Div(code, title=f"{species} — {stimuli.LABEL_DEF[label]['name']}",
                    style={"backgroundColor": color, "color": "#000",
                           "borderRadius": "8px", "padding": "14px 10px", "textAlign": "center",
                           "fontWeight": "bold", "fontFamily": "Consolas, monospace",
                           "minWidth": "56px", "boxShadow": "0 1px 3px rgba(0,0,0,.4)"})


def _legend():
    items = []
    for lab, d in stimuli.LABEL_DEF.items():
        items.append(html.Div([
            html.Span(style={"display": "inline-block", "width": "14px", "height": "14px",
                             "backgroundColor": d["color"], "borderRadius": "3px", "marginRight": "6px",
                             "verticalAlign": "middle"}),
            html.Span(f"{lab} — {d['name']}", style={"fontSize": "12px", "color": INK}),
        ], style={"marginRight": "16px", "display": "inline-block"}))
    return html.Div(items, style={"marginTop": "8px"})


def _stimulus_grid():
    header = [html.Th("", style={"padding": "6px"})] + [
        html.Th(lab, style={"color": stimuli.label_color(lab), "padding": "6px"}) for lab in stimuli.LABEL_DEF
    ]
    rows = []
    for sp in stimuli.STIM_SPECIES:
        cells = [html.Td(sp, style={"color": INK, "fontWeight": "bold", "padding": "6px"})]
        for lab in stimuli.LABEL_DEF:
            cells.append(html.Td(_chip(sp, lab), style={"padding": "4px"}))
        rows.append(html.Tr(cells))
    return html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)],
                      style={"borderCollapse": "collapse"})


def _partition_table():
    labels = list(stimuli.LABEL_DEF)
    header = [html.Th("Run", style={"padding": "6px", "color": MUTED})] + [
        html.Th(lab, style={"padding": "6px", "color": stimuli.label_color(lab)}) for lab in labels]
    part_color = {0: "#3a3a3a", 1: "#2f5fa0", 2: "#7a4fa0"}
    rows = []
    for run in stimuli.RUNS:
        cells = [html.Td(run, style={"padding": "6px", "color": INK,
                                     "fontFamily": "Consolas, monospace"})]
        for lab in labels:
            p = stimuli.PARTITIONS[run][lab]
            cells.append(html.Td(str(p), style={"padding": "6px", "textAlign": "center",
                                                 "backgroundColor": part_color[p], "color": "white"}))
        rows.append(html.Tr(cells))
    return html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)],
                      style={"borderCollapse": "collapse", "marginTop": "4px"})


app.layout = html.Div(style={"backgroundColor": DARK_BG, "color": INK, "minHeight": "100vh",
                             "padding": "12px 16px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("Analysis Planner", style={"textAlign": "center", "margin": "4px 0 10px", "color": INK}),

    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "12px 16px",
                    "marginBottom": "10px"}, children=[
        html.H4("EmoC stimulus conditions", style={"margin": "0 0 8px", "color": INK}),
        html.Div("2 stimulus species × 5 emotion labels = 10 conditions. These are the cells an "
                 "RSA model or GLM contrast operates on. Icons will replace the text codes later.",
                 style={"fontSize": "12px", "color": MUTED, "marginBottom": "10px"}),
        _stimulus_grid(),
        _legend(),
    ]),

    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "12px 16px",
                    "marginBottom": "10px"}, children=[
        html.H4("Partition design", style={"margin": "0 0 4px", "color": INK}),
        html.Div("0 = non-repeated · 1 = partition 1 · 2 = partition 2 (runs 5–6).",
                 style={"fontSize": "12px", "color": MUTED, "marginBottom": "6px"}),
        _partition_table(),
    ]),

    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "12px 16px"}, children=[
        html.H4("Design an analysis", style={"margin": "0 0 6px", "color": INK}),
        html.Div("Coming in a later phase: drag conditions into groups to build an RSA dissimilarity "
                 "model or define GLM contrasts, preview the matrix, then send it straight to the "
                 "Scheduler tab. For now use the RSA Builder and Scheduler tabs.",
                 style={"fontSize": "13px", "color": INK}),
    ]),
])


if __name__ == "__main__":
    app.run(debug=True, port=8053)

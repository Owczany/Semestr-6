import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODERN_TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#2563eb"
SECONDARY_COLOR = "#0f766e"
ACCENT_COLOR = "#7c3aed"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e5e7eb"


def _label_to_letter(label):
    return {i: chr(ord("A") + i) for i in range(26) if i != 9}.get(label, str(label))


def _apply_modern_layout(fig, title, height=420):
    fig.update_layout(
        template=MODERN_TEMPLATE,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        margin=dict(l=48, r=24, t=72, b=48),
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#111827"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID_COLOR,
        tickfont=dict(color=MUTED_COLOR),
        title_font=dict(color=MUTED_COLOR),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=GRID_COLOR,
        tickfont=dict(color=MUTED_COLOR),
        title_font=dict(color=MUTED_COLOR),
    )
    return fig


def plot_class_distribution(df, label_column="label", title="Class distribution"):
    counts = df[label_column].value_counts().sort_index()
    labels = [_label_to_letter(label) for label in counts.index]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker_color=PRIMARY_COLOR,
            marker_line_width=0,
            hovertemplate="Class %{x}<br>Images: %{y}<extra></extra>",
        )
    )
    _apply_modern_layout(fig, title)
    fig.update_xaxes(title_text="Class")
    fig.update_yaxes(title_text="Number of images")
    fig.show()
    return fig


def plot_label_samples(
    df,
    samples_per_class=5,
    label_column="label",
    title="Examples from each class",
):
    classes = sorted(df[label_column].unique())
    fig = make_subplots(
        rows=len(classes),
        cols=samples_per_class,
        vertical_spacing=0.003,
        horizontal_spacing=0.003,
    )

    for row, label in enumerate(classes, start=1):
        samples = df[df[label_column] == label].iloc[:samples_per_class]
        for col in range(1, samples_per_class + 1):
            pixels = samples.iloc[col - 1, 1:].values.reshape(28, 28).astype(np.uint8)
            fig.add_trace(
                go.Heatmap(
                    z=pixels,
                    colorscale="gray",
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, autorange="reversed", row=row, col=col)

        fig.add_annotation(
            text=_label_to_letter(label),
            xref="paper",
            yref="paper",
            x=-0.015,
            y=1 - ((row - 0.5) / len(classes)),
            showarrow=False,
            font=dict(size=12, color=MUTED_COLOR),
        )

    _apply_modern_layout(fig, title, height=max(900, math.ceil(len(classes) * 52)))
    fig.update_layout(showlegend=False, margin=dict(l=64, r=24, t=72, b=24))
    fig.show()
    return fig


def plot_training_history(history, title):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss", "Accuracy"),
        horizontal_spacing=0.11,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="Train loss",
            line=dict(color=PRIMARY_COLOR, width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines+markers",
            name="Val loss",
            line=dict(color=SECONDARY_COLOR, width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_acc"],
            mode="lines+markers",
            name="Train acc",
            line=dict(color=ACCENT_COLOR, width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_acc"],
            mode="lines+markers",
            name="Val acc",
            line=dict(color=MUTED_COLOR, width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=2,
    )

    _apply_modern_layout(fig, title, height=460)
    fig.update_xaxes(title_text="Epoch", dtick=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
    fig.show()
    return fig

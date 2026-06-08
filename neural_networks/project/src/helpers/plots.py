import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODERN_TEMPLATE = "plotly_white"
PLOT_FONT = "Segoe UI, Arial, Helvetica, sans-serif"
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
        font=dict(family=PLOT_FONT, size=14, color="#111827"),
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
    return fig


def plot_label_samples(
    df,
    samples_per_class=3,
    label_column="label",
    title="Examples from each class",
):
    classes = sorted(df[label_column].unique())
    fig, axes = plt.subplots(
        len(classes),
        samples_per_class,
        figsize=(samples_per_class * 1.8, len(classes) * 1.45),
        squeeze=False,
    )

    for row, label in enumerate(classes):
        samples = df[df[label_column] == label].iloc[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col >= len(samples):
                ax.axis("off")
                continue

            pixels = samples.iloc[col, 1:].values.reshape(28, 28).astype(np.uint8)

            ax.imshow(
                pixels,
                cmap="gray",
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if col == 0:
                ax.set_ylabel(
                    _label_to_letter(label),
                    rotation=0,
                    labelpad=18,
                    va="center",
                    ha="right",
                    fontsize=12,
                    color=MUTED_COLOR,
                    fontfamily="Arial",
                )

    fig.suptitle(title, fontsize=16, fontfamily="Arial", color="#111827", y=0.995)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.97,
        bottom=0.01,
        wspace=0.08,
        hspace=0.08,
    )
    plt.close(fig)
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
    return fig


def plot_experiment_comparison(
    results_df,
    metric="best_val_acc",
    title="Experiment comparison",
):
    hover_df = results_df.copy()
    if "model" not in hover_df.columns:
        hover_df["model"] = ""

    fig = go.Figure(
        go.Bar(
            x=results_df["name"],
            y=results_df[metric],
            marker_color=PRIMARY_COLOR,
            marker_line_width=0,
            customdata=hover_df[
                ["model", "optimizer", "learning_rate", "momentum", "best_epoch"]
            ].to_numpy(),
            hovertemplate=(
                "%{x}<br>"
                "Best val acc: %{y:.4f}<br>"
                "Model: %{customdata[0]}<br>"
                "Optimizer: %{customdata[1]}<br>"
                "LR: %{customdata[2]}<br>"
                "Momentum: %{customdata[3]}<br>"
                "Best epoch: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    _apply_modern_layout(fig, title)
    fig.update_xaxes(title_text="Experiment")
    fig.update_yaxes(title_text=metric, range=[0, 1])
    return fig

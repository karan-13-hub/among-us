import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

def plot_roc_curve_eval(labels, probe_outputs, labels_2=None, names=None):
    """
    Plot ROC curve for probe outputs and labels.
    
    Args:
        labels: Ground truth binary labels
        probe_outputs: Predicted probabilities from probe
        labels_2: Optional second set of ground truth binary labels
        names: Optional list of 2 strings for legend labels
    """
    # Among Us color theme
    among_us_colors = [
        ("#c51111", "#7a0838"),  # Red
        ("#132ed1", "#09158e"),  # Blue
        ("#117f2d", "#0a4d2e"),  # Green
        ("#ed54ba", "#ab2bad"),  # Pink
        ("#ef7d0d", "#b33e15"),  # Orange
        ("#f5f543", "#c28722"),  # Yellow
        ("#6b2fbb", "#3b177c"),  # Purple
    ]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probe_outputs)
    roc_auc = auc(fpr, tpr)

    # Create ROC plot
    fig = go.Figure()
    
    # Set name for first curve based on provided names
    curve_name = f'ROC (AUC = {roc_auc:.3f})' if names is None else f'{names[0]} (AUC = {roc_auc:.3f})'
    
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            mode='lines', 
                            name=curve_name,
                            line=dict(color=among_us_colors[1][0], width=2.5)))
    
    # Add second curve if labels_2 is provided
    if labels_2 is not None:
        fpr_2, tpr_2, _ = roc_curve(labels_2, probe_outputs)
        roc_auc_2 = auc(fpr_2, tpr_2)
        
        # Set name for second curve based on provided names
        curve_name_2 = f'ROC (AUC = {roc_auc_2:.3f})' if names is None or len(names) < 2 else f'{names[1]} (AUC = {roc_auc_2:.3f})'
        
        fig.add_trace(go.Scatter(x=fpr_2, y=tpr_2,
                                mode='lines',
                                name=curve_name_2,
                                line=dict(color=among_us_colors[0][0], width=2.5)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(dash='dash', color='gray')))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )

    # ticks on both axes
    fig.update_xaxes(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.update_yaxes(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])

    fig.update_layout(
        title='',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Computer Modern', size=16),
        xaxis=dict(title='False Positive Rate', gridcolor='lightgray', showgrid=True, zeroline=True, range=(0,1)),
        yaxis=dict(title='True Positive Rate', gridcolor='lightgray', showgrid=True, zeroline=True, range=(0,1)),
        legend=dict(x=0.32, y=0.05, bgcolor='rgba(255, 255, 255, 0.8)'),
        # margin=dict(l=60, r=20, t=20, b=60),
        width=500, height=500
    )

    if labels_2 is None:
        return fig, roc_auc
    else:
        return fig, [roc_auc, roc_auc_2]

# Print performance metrics for each group
def print_metrics(data, group_name, threshold=0.5, behaviors=None):
    print(f"\nMetrics for {group_name}:")
    for behavior in behaviors:
        truth = data[f'{behavior}_truth']
        pred = (data['probe_output'] > threshold).astype(int)
        
        accuracy = accuracy_score(truth, pred)
        precision = precision_score(truth, pred)
        recall = recall_score(truth, pred)
        f1 = f1_score(truth, pred)
        
        print(f"{behavior:<20} acc:{accuracy:>7.3f} pre:{precision:>7.3f} rec:{recall:>7.3f} f1:{f1:>7.3f}")

def add_roc_curves(data, col_num, fig, behaviors, colors):
    """
    Add ROC curves to a plotly figure for different behaviors.
    
    Args:
        data (pd.DataFrame): DataFrame containing behavior data
        col_num (int): Column number for subplot
        fig (go.Figure): Plotly figure to add traces to
        behaviors (list): List of behavior names to plot
        colors (list): List of colors for each behavior curve
    """
    # Create a copy of the data to avoid SettingWithCopyWarning
    data_copy = data.copy()
    
    for behavior, color in zip(behaviors, colors):
        # Use .loc to avoid SettingWithCopyWarning
        truth = data_copy.loc[:, f'{behavior}_truth']
        pred = data_copy.loc[:, 'probe_output']
        
        # Skip if no positive or negative samples
        if truth.sum() == 0 or (1 - truth).sum() == 0:
            continue
            
        fpr, tpr, _ = roc_curve(truth, pred)
        auc_score = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, 
                      name=f'{behavior} (AUC = {auc_score:.3f})',
                      line=dict(color=color),
                      showlegend=True,
                      legendgroup=str(col_num),
                      legendgrouptitle_text=f'Plot {col_num}'),
            row=1, col=col_num
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1],
                      line=dict(color='black', dash='dash'),
                      showlegend=False,
                      legendgroup=str(col_num)),
            row=1, col=col_num
        )
        
        # Configure legend for this subplot
        fig.update_layout(**{
            f'legend{col_num}': dict(
                yanchor="top",
                y=1.0,
                xanchor="left", 
                x=0.05 + (col_num-1)*0.33,
                orientation="v"
            )
        })

def plot_roc_curves(probe_eval, title="ROC Curves for Different Behaviors (all actions)", behaviors=None):
    """
    Create ROC curve plots for different player groups and behaviors.
    
    Args:
        probe_eval (pd.DataFrame): DataFrame containing probe evaluation data
        title (str): Title for the plot
        
    Returns:
        go.Figure: Plotly figure object with ROC curves
    """
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('All Players','Impostors Only', 'Crewmates Only'),
                        shared_yaxes=True)

    # Colors for different behaviors
    colors = ['blue', 'red', 'green', 'orange']

    # Add ROC curves for all groups
    add_roc_curves(probe_eval, 1, fig, behaviors, colors)
    add_roc_curves(probe_eval[probe_eval['player_identity'] == 'Impostor'], 2, fig, behaviors, colors)
    add_roc_curves(probe_eval[probe_eval['player_identity'] == 'Crewmate'], 3, fig, behaviors, colors)

    # Update layout
    fig.update_layout(
        height=350,
        width=900,
        title_text=title,
        showlegend=True
    )

    # Update axes labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=i)
        if i == 1:
            fig.update_yaxes(title_text="True Positive Rate", row=1, col=i)

    fig.update_layout({'plot_bgcolor': 'rgba(255, 255, 255, 1)',})
    # show fine grid lines on both axes on both subplots
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # legend inside the plot in a box
    fig.update_layout(legend=dict(x=1, y=1, bgcolor="white"))

    # everthing latex font (for research paper)
    fig.update_layout(font=dict(family='serif', size=15, color='black'))
    fig.update_xaxes(title_font=dict(family='serif', size=18, color='black'))
    fig.update_yaxes(title_font=dict(family='serif', size=18, color='black'))
    fig.update_xaxes(tickfont=dict(family='serif', size=18, color='black'))
    fig.update_yaxes(tickfont=dict(family='serif', size=18, color='black'))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
    
    return fig

def plot_behavior_distribution(summary_df):
    """
    Creates a grouped bar plot showing behavior distribution between Impostors and Crewmates.
    
    Args:
        summary_dfs: List of summary dataframes
        index: Index of summary dataframe to use (default 2)
    
    Returns:
        plotly.graph_objects.Figure
    """

    summary_df['awareness'] = summary_df['awareness'].astype(int)
    summary_df['lying'] = summary_df['lying'].astype(int) 
    summary_df['deception'] = summary_df['deception'].astype(int)
    summary_df['planning'] = summary_df['planning'].astype(int)

    behaviors = ['awareness', 'lying', 'deception', 'planning']
    impostor_df = summary_df[summary_df['player_identity'] == 'Impostor'].copy()
    crewmate_df = summary_df[summary_df['player_identity'] == 'Crewmate'].copy()

    # Get percentages and counts for both impostors and crewmates
    impostor_pcts = {}
    crewmate_pcts = {}
    impostor_counts = {}
    crewmate_counts = {}
    for behavior in behaviors:
        impostor_high = impostor_df[impostor_df[behavior] > 5].shape[0]
        crewmate_high = crewmate_df[crewmate_df[behavior] > 5].shape[0]
        impostor_pcts[behavior] = (impostor_high / impostor_df.shape[0]) * 100
        crewmate_pcts[behavior] = (crewmate_high / crewmate_df.shape[0]) * 100
        impostor_counts[behavior] = impostor_high
        crewmate_counts[behavior] = crewmate_high

    # Create dataframe with both percentages
    df = pd.DataFrame({
        'Behavior': behaviors + behaviors,
        'Percentage': list(impostor_pcts.values()) + list(crewmate_pcts.values()),
        'Count': list(impostor_counts.values()) + list(crewmate_counts.values()),
        'Role': ['Impostor']*len(behaviors) + ['Crewmate']*len(behaviors)
    })

    # Create grouped bar plot with red for impostor, blue for crewmate
    fig = px.bar(df, x='Behavior', y='Percentage', color='Role',
                 barmode='group',
                 color_discrete_map={'Impostor': 'Red', 'Crewmate': 'Blue'})

    # Add count labels on top of bars
    for i in range(len(fig.data)):
        fig.add_traces(go.Scatter(
            x=fig.data[i].x,
            y=fig.data[i].y,
            text=df[df['Role'] == fig.data[i].name]['Count'],
            mode='text',
            textposition='top left' if fig.data[i].name == 'Impostor' else 'top right',
            showlegend=False,
            textfont=dict(family='serif', size=15, color='black')
        ))

    fig.update_layout({'plot_bgcolor': 'rgba(255, 255, 255, 1)',})
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.update_layout(width=600, height=500)
    fig.update_yaxes(title_text='Percentage')

    # everything latex font (for research paper)
    fig.update_layout(font=dict(family='serif', size=15, color='black'))
    fig.update_xaxes(title_font=dict(family='serif', size=18, color='black'))
    fig.update_yaxes(title_font=dict(family='serif', size=18, color='black'))
    fig.update_xaxes(tickfont=dict(family='serif', size=18, color='black'))
    fig.update_yaxes(tickfont=dict(family='serif', size=18, color='black'))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)

    return fig

def plot_metrics(metrics: dict, title: str = "Training Progress", x_label: str = "Epoch", y_label: str = "Metric", y_range: tuple = (0, 1)):
    """
    Plots multiple line graphs on the same figure.
    
    Args:
        metrics (dict): Dictionary where keys are labels and values are lists of y-values.
        title (str): Title of the plot.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        y_range (tuple): y-axis range (default: (0,1)).
    """
    fig = go.Figure()
    
    for label, y_values in metrics.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(y_values))),
            y=y_values,
            name=label,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Computer Modern', size=16),
        xaxis=dict(title=x_label, gridcolor='lightgray', showgrid=True, zeroline=False),
        yaxis=dict(title=y_label, gridcolor='lightgray', showgrid=True, zeroline=False, range=y_range),
        legend=dict(x=0.62, y=0.22, bgcolor='rgba(255, 255, 255, 0.8)'),
        margin=dict(l=60, r=20, t=20, b=60),
        width=500, height=400
    )

    return fig
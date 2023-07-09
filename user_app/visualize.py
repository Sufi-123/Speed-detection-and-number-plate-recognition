import matplotlib.pyplot as plt
from django.conf import settings
import os

def generate_bar_graph(exceeded_limit, within_limit):
    # Get the counts for vehicles that have crossed the speed limit and those that have not
    exceeded_limit_count = len(exceeded_limit)
    within_limit_count = len(within_limit)

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the bar chart
    ax.bar(['Exceeded Speed Limit', 'Within Speed Limit'], [exceeded_limit_count, within_limit_count], color=['red', 'green'])

    # Add labels and title
    ax.set_xlabel('Speed Limit Status')
    ax.set_ylabel('Number of Vehicles')
    ax.set_title('Vehicle Speed Limit')

    # Save the chart to a file
    chart_path = os.path.join(settings.STATIC_ROOT, 'chart.png')  # Specify the file path to save the chart image
    plt.savefig(chart_path)  # Save the chart image to the file

    return chart_path


def generate_line_graph(speeds):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the line graph
    ax.plot(range(1, len(speeds) + 1), speeds)

    # Add labels and title
    ax.set_xlabel('Record')
    ax.set_ylabel('Speed')
    ax.set_title('Speed Record')

    # Save the chart to a file
    graph_path = '../static/graph.png'  # Specify the file path to save the chart image
      # Save the chart image to the file

    return plt.savefig(graph_path)

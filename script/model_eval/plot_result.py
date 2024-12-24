import numpy as np
import matplotlib.pyplot as plt

class PlotResult:

    def __init__(self):
        pass

    def plot_result(self, y, models, result, filename):
        rmse_values = []
        upper_percentile = []
        lower_percentile = []

        for result_map in result:
            rmse_values.append(round(result_map[y]['RMSE']))
            lower_percentile.append(round(result_map[y]['RMSE 95% CI'][0]))
            upper_percentile.append(round(result_map[y]['RMSE 95% CI'][1]))

        # Convert to errors for matplotlib (asymmetric error)
        lower_error = np.array(rmse_values) - np.array(lower_percentile)
        upper_error = np.array(upper_percentile) - np.array(rmse_values)

        # Combine lower and upper errors for plotting
        asymmetric_error = [lower_error, upper_error]

        # Create a bar chart
        plt.figure(figsize=(14, 12))
        bars = plt.bar(models, rmse_values, yerr=asymmetric_error, capsize=5, color='skyblue', label='RMSE')

        # Add labels and title
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Values of {y} with Different Models')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

        # Add percentile labels at the top and bottom of the error bars
        for i, (lower, upper) in enumerate(zip(lower_percentile, upper_percentile)):
            plt.text(i, upper, f'{upper}', ha='center', va='bottom', color='green')  # Upper percentile label
            plt.text(i, lower, f'{lower}', ha='center', va='top', color='red')  # Lower percentile label

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

        # Add legend specifying the color coding
        plt.legend(handles=[bars], labels=['RMSE with 5-95% CI'], loc='upper right')

        # Show the plot
        plt.show()

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
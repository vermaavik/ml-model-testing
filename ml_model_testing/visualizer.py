import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction_distributions(v1, v2):
    sns.kdeplot(v1, label='Model V1')
    sns.kdeplot(v2, label='Model V2')
    plt.title("Prediction Distribution Comparison")
    plt.legend()
    plt.show()

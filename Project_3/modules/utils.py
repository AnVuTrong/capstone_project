from matplotlib import pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, model_name):
	plt.figure(figsize=(10, 7))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
	            xticklabels=["negative", "neutral", "positive"],
	            yticklabels=["negative", "neutral", "positive"])
	plt.title(f'Confusion Matrix for {model_name}')
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.show()

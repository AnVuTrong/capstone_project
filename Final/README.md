# How to review this project
***
## 1. Introduction
- Project 1 using traditional machine learning to clustering segment and grouping the customers based on their purchase history.
- The project structure consists of:
<details>
  <summary>Project's structure:</summary>

```
Final
    powerpoint.pptx
    README.md
    .streamlit
    ├── config.toml (GUI color setting)
    
    GUI
    ├── style.css (font style for the GUI)
    ├── streamlit_ui.py (main file of the GUI)
    ├── img
    │   ├── All the images used in the GUI
    ├── modules
    │   ├── backend
    │   │   ├── recommendation_system.py (Handle the calculation of the recommendation system of project 2)
    │   ├── pages
    │   │   ├── homepage.py (Homepage of the GUI)
    │   │   ├── about_us.py (About us page)
    │   │   ├── how_it_works.py (How it works page)
    │   │   ├── customer_segmentation.py (Project 1 main page)
    │   │   ├── collaborative_filtering.py (Project 2 page)
    │   │   ├── content_based_filtering.py (Project 2 page)
    │   ├── sidebar.py (Work as the main nevigation bar)
    │   ├── page_manager.py (Handle the page navigation)
    │   ├── widgets.py (Handle the custom widgets of the GUI)

    Project_1
    ├── data
    │   ├── raw (Products_with_Prices.csv, Transactions.csv)
    │   └── demo (e-commerce.csv, demo_e_commerce.ipynb)
    ├── modules
    │   ├── data_preprocessing.py (Handle the data preprocessing of project 1)
    │   ├── machine_learning.py (Handle the machine learning of project 1)
    │   ├── feature_engineering.py (Handle the feature engineering of project 1)
    │   ├── data_analysis.py (Handle the data analysis of project 1)
    │   ├── exploratory_data_analyst.py (Handle the exploratory data analyst of project 1)
    ├── notebooks
    │   ├── pandas_previewer.ipynb (Preview the data of project 1, only the first step)
    ├── small_scale_data_analyst_and_machine_learning.ipynb (Project 1 main file)
    ├── large_scale_big_data.ipynb (Project 1 main file)
```
</details>

## 2. Project 1: How to review the project
There are two files to be reviewed:
- **small_scale_data_analyst_and_machine_learning.ipynb**

The small-scale file is using the traditional machine learning to clustering segment
and grouping the customers based on their purchase history.
  (Sklearn, Pandas, Numpy, Matplotlib, Seaborn) 

- **large_scale_big_data.ipynb**

The large-scale file is using the big data technology to clustering segment and grouping the customers based on their purchase history
User may use whichever file to review the project based on their preference.
(Pyspark, Pandas, Numpy, Matplotlib, Seaborn)

## 3. Project 1: How to run the project
- **small_scale_data_analyst_and_machine_learning.ipynb**

***First we need to import the necessary libraries:***
```python
from modules.data_preprocessing import PandasDataPreprocessing
from modules.exploratory_data_analyst import PandasEDA
from modules.data_analysis import PandasCustomerSegmentation as PandasCS
from modules.feature_engineering import PandasFeatureEngineering as PandasFE
from modules.machine_learning import PandasMachineLearning as PandasML
```

***Then we need to load the data:***
```python
preprocessor = PandasDataPreprocessing(
    'data/Products_with_Prices.csv',
    'data/Transactions.csv',
)
df_transaction, df_rfm = preprocessor.run()
```

***Next we need to do the exploratory data analysis:***
```python
# EDA for RFM
eda_rfm = PandasEDA(df_rfm)
eda_rfm.run_all()

# EDA for transaction
eda_trans = PandasEDA(df_transaction)
eda_trans.plot_distributions()
```

***Then we need to do the feature engineering:***
```python
feature_engineering = PandasFE(df_rfm)
feature_engineering.run()
df_ml = df_rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment_Encoded']]
```

***Finally we need to do the machine learning:***
```python 
# Evaluate the models
pandas_ml = PandasML(df_ml, number_of_clusters=3)
results_df = pandas_ml.evaluate_models()
pandas_ml.visualize_model_evaluation(results_df)
kmeans_models, silhouette_scores = pandas_ml.elbow_method_kmeans()
kmeans_model = kmeans_models[5] (5 is the best number of clusters)

# cluster the data
df_ml['Cluster'] = kmeans_model.labels_

# visualize the clusters
pandas_ml.visualize_kmeans_clusters(5)
pandas_ml.visualize_hierarchical_clusters()
```


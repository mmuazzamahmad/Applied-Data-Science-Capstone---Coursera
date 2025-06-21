# SpaceX Falcon 9 First Stage Landing Prediction

## 1. Project Overview

This repository contains the full data science capstone project focused on predicting whether the Falcon 9 first stage will land successfully. SpaceX's ability to reuse the first stage of its rockets has dramatically reduced the cost of launches, making this a critical factor in the space industry. A successful landing can significantly impact the cost of a launch, dropping it from upwards of $165 million to around $62 million.

This project walks through the entire data science pipeline:
- **Data Collection**: Gathering data from the SpaceX REST API and a Wikipedia page.
- **Data Wrangling & Preprocessing**: Cleaning, structuring, and preparing the data for analysis.
- **Exploratory Data Analysis (EDA)**: Using SQL and data visualization to uncover insights and relationships.
- **Interactive Visualization**: Building interactive maps with Folium to analyze launch site locations.
- **Machine Learning**: Creating a predictive model using various classification algorithms to predict landing success.
- **Dashboard Creation**: Building an interactive dashboard with Plotly Dash to visualize the results.

---

## 2. Project Structure

This project is divided into several parts, each contained in its own Jupyter Notebook or Python script. They are designed to be run in the following order:

1.  **`Data Collection API.ipynb`**: Collects data via the SpaceX API.
2.  **`Data Collection with Web Scraping.ipynb`**: Gathers additional data from Wikipedia using web scraping.
3.  **`Data Wrangling.ipynb`**: Cleans the data and creates the target variable for our model.
4.  **`EDA with SQL.ipynb`**: Performs exploratory data analysis using SQL queries.
5.  **`EDA with Data Visualization.ipynb`**: Creates visualizations to understand the data and performs feature engineering.
6.  **`Interactive Visual Analytics with Folium.ipynb`**: Analyzes launch site locations with interactive maps.
7.  **`Machine Learning Prediction.ipynb`**: Builds, trains, and evaluates multiple classification models.
8.  **`spacex_dash_app.py`**: An interactive dashboard application built with Plotly Dash.

---

## 3. Methodology & File Descriptions

### 3.1. `Data Collection API.ipynb`
- **Objective**: Collect initial launch data using the SpaceX v4 API.
- **Process**:
    - Makes HTTP GET requests to the SpaceX API endpoint for past launches.
    - Parses the JSON response into a Pandas DataFrame.
    - Enriches the initial dataset by making further API calls for details on rockets, launchpads, payloads, and cores.
    - Performs initial data cleaning by filtering for single-core, single-payload Falcon 9 launches.
    - Handles missing `PayloadMass` values by imputing the mean.
- **Output**: `dataset_part_1.csv`

### 3.2. `Data Collection with Web Scraping.ipynb`
- **Objective**: Scrape a comprehensive table of Falcon 9 and Falcon Heavy launch records from Wikipedia.
- **Process**:
    - Uses `requests` and `BeautifulSoup` to fetch and parse the HTML from a static Wikipedia page.
    - Identifies and extracts the main launch records table.
    - Iterates through the table rows (`<tr>`) and data cells (`<td>`) to extract information such as flight number, date, booster version, payload, and launch outcome.
    - Populates a dictionary with the scraped data and converts it into a Pandas DataFrame.
- **Output**: `spacex_web_scraped.csv`

### 3.3. `Data Wrangling.ipynb`
- **Objective**: Create the binary classification target variable (`Class`) from the collected data.
- **Process**:
    - Loads the dataset from `dataset_part_1.csv`.
    - Analyzes the `Outcome` column, which contains various landing statuses (e.g., `True ASDS`, `False Ocean`).
    - Defines a set of "bad outcomes" (failures or no attempts).
    - Creates a new `Class` column where `1` represents a successful landing and `0` represents an unsuccessful one.
- **Output**: `dataset_part_2.csv`

### 3.4. `EDA with SQL.ipynb`
- **Objective**: Perform exploratory data analysis using SQL.
- **Process**:
    - Uses the `ipython-sql` extension to run SQL queries directly within the notebook.
    - Connects to a DB2 database where the dataset has been loaded.
    - **Queries Executed**:
        - Display unique launch sites.
        - Filter for records from launch sites starting with 'CCA'.
        - Calculate the total payload mass for NASA (CRS) missions.
        - Find the average payload mass for the F9 v1.1 booster version.
        - Identify the date of the first successful ground pad landing.
        - List boosters with successful drone ship landings and payloads between 4000-6000 kg.
        - Rank landing outcomes by frequency between 2010 and 2017.

### 3.5. `EDA with Data Visualization.ipynb`
- **Objective**: Visualize the data to identify key trends and perform feature engineering.
- **Process**:
    - Uses `Matplotlib` and `Seaborn` to create several plots:
        - **Flight Number vs. Payload Mass**: Shows that heavier payloads are attempted on later flights.
        - **Flight Number vs. Launch Site**: Illustrates the usage frequency and success trends of different launch sites over time.
        - **Payload Mass vs. Launch Site**: Visualizes payload distribution for each site.
        - **Success Rate by Orbit Type**: A bar chart showing which orbits have higher success rates.
    - **Feature Engineering**:
        - Selects relevant features for the machine learning model.
        - Applies one-hot encoding to categorical columns (`Orbit`, `LaunchSite`, `LandingPad`, `Serial`) using `pd.get_dummies()`.
- **Output**: `dataset_part_3.csv` (The feature-engineered dataset for modeling).

### 3.6. `Interactive Visual Analytics with Folium.ipynb`
- **Objective**: Analyze the geographical properties of launch sites using interactive maps.
- **Process**:
    - Uses the `Folium` library to create an interactive map.
    - **Task 1**: Marks all launch sites with circles and labels.
    - **Task 2**: Adds success/failure markers for each launch at its respective site, using a `MarkerCluster` to handle overlapping points. Green markers indicate success (`class=1`) and red markers indicate failure (`class=0`).
    - **Task 3**: Calculates and visualizes the distances from launch sites to nearby points of interest like coastlines, highways, and railways using `PolyLine` objects.

### 3.7. `Machine Learning Prediction.ipynb`
- **Objective**: Build and evaluate machine learning models to predict landing success.
- **Process**:
    1.  **Data Preparation**: Loads the feature-engineered data (`dataset_part_3.csv`) and the target variable (`Y`).
    2.  **Standardization**: Standardizes the feature set `X` using `preprocessing.StandardScaler()`.
    3.  **Data Splitting**: Splits the data into training and testing sets (80/20 split).
    4.  **Model Training and Tuning**:
        - Implements Logistic Regression, Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbors (KNN) classifiers.
        - Uses `GridSearchCV` with 10-fold cross-validation to find the optimal hyperparameters for each model.
    5.  **Model Evaluation**:
        - Calculates the accuracy, Jaccard score, and F1-score for each model on the test data.
        - Plots a confusion matrix for each model to analyze its performance.

### 3.8. `spacex_dash_app.py`
- **Objective**: Create an interactive web dashboard for data exploration.
- **Process**:
    - Builds a dashboard using `Plotly Dash`.
    - **Components**:
        - A dropdown menu to filter data by launch site.
        - A range slider to filter data by payload mass.
    - **Visualizations**:
        - A pie chart that dynamically updates to show the success rate for the selected launch site (or all sites).
        - A scatter plot showing the relationship between payload mass and launch success, colored by booster version, which also updates based on the filters.

---

## 4. Key Findings & Results

### Exploratory Data Analysis
- **Launch Site Proximity**: All launch sites are located near the equator to leverage Earth's rotational speed and are close to coastlines to minimize risk to populated areas during launch.
- **Success Rate Trend**: The success rate of launches has generally increased over the years, indicating a learning curve and improvements in technology.
- **Orbit Success**: Certain orbits like `SSO`, `GEO`, and `ES-L1` have a 100% success rate in this dataset, while others like `GTO` have a lower success rate, often associated with heavier payloads.

### Machine Learning Performance
All models performed well, achieving an accuracy of 83.3% on the test set. The Decision Tree Classifier slightly outperformed the others in terms of overall accuracy and F1-score on the complete dataset.

| Model | Jaccard Score (Test) | F1-Score (Test) | Accuracy (Test) |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.80 | 0.89 | 0.83 |
| SVM | 0.80 | 0.89 | 0.83 |
| Decision Tree | 0.80 | 0.89 | 0.83 |
| KNN | 0.80 | 0.89 | 0.83 |

*Note: The identical performance on the test set is likely due to the small size of the test data (18 samples). The Decision Tree model demonstrated the highest accuracy (91.1%) when evaluated against the entire dataset.*

---

## 5. How to Run the Project

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- requests
- beautifulsoup4
- folium
- wget
- plotly
- dash
- ipython-sql
- sqlalchemy
- ibm_db_sa

### Instructions
1.  Clone the repository.
2.  Install the required dependencies using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn requests beautifulsoup4 folium wget plotly dash ipython-sql sqlalchemy==1.3.9 ibm_db_sa
    ```
3.  Run the Jupyter Notebooks in the specified order to reproduce the data processing and analysis pipeline.
4.  To launch the interactive dashboard, run the Python script from your terminal:
    ```bash
    python spacex_dash_app.py
    ```
    Then, navigate to the local URL provided in the terminal (usually `http://127.0.0.1:8050/`).

---

## 6. Credits
This project was completed as part of the IBM Data Science Professional Certificate capstone. The notebooks and project structure are based on the materials provided by IBM and Coursera.

# **Permeability Prediction using Machine Learning**

## **Project Overview**

Permeability is a critical property in various fields, from oil and gas extraction to environmental monitoring. It measures a material's ability to allow fluids to flow through it. Accurately predicting permeability is essential for optimizing extraction processes, analyzing fluid migration, and modeling porous media systems.

This project applies **machine learning algorithms** to predict the permeability of rock samples using a variety of physical and structural characteristics. By leveraging advanced algorithms, we aim to improve prediction accuracy over traditional experimental methods, which are often time-consuming and expensive.

## **Dataset Description**

The dataset comprises **266 mini-samples** derived from digital models of real and synthetic rocks, scanned using **micro-computed tomography (micro-CT)**. Each sample contains the following features:
- **Pore Radius (μm)** (`rp`): The average radius of pores in the sample.
- **Throat Radius (μm)** (`rt`): The average radius of pore throats.
- **Coordination Number** (`Nc`): The number of connections a pore has with other pores.
- **Porosity** (`ϕ`): The fraction of the rock volume occupied by pores.
- **Specific Surface Area (1/μm)** (`Ss`): The surface area per unit volume of the porous space.
- **Tortuosity** (`τ`): A measure of how convoluted the pathways through the pores are.
- **Permeability (μm²)** (`k`): The target variable, representing the sample's permeability.

These mini-samples were segmented from larger rock samples to increase the variety of data points, ensuring that the models can generalize across different types of porous media, including **carbonate and sandstone** rocks.

### **Data Sources**

The digital models and scans of the rock samples were sourced from:
- **Professor M. Blunt's research group** at Imperial College London (Open-access rock datasets)
- The models were segmented and processed to extract meaningful features for machine learning.

## **Project Goals**

- **Primary Objective:** To predict the permeability of rocks using machine learning models with high accuracy, reducing reliance on expensive laboratory-based permeability measurements.
- **Secondary Objectives:** 
  - Compare the performance of various machine learning algorithms.
  - Identify the most significant features influencing permeability.
  - Optimize training and testing datasets for the most accurate predictions.

## **Machine Learning Algorithms**

To achieve accurate permeability predictions, the following machine learning algorithms were employed:

1. **Random Forest (RF)**: A robust ensemble method that creates multiple decision trees and aggregates their results.
2. **Gradient Boosting (GB)**: A sequential learning algorithm that improves the model iteratively by correcting errors of previous models.
3. **Linear Regression (LR)**: A regression model that uses support vectors to define a hyperplane for predicting outcomes.

### **Model Performance**

- **Random Forest (RF)** emerged as the best-performing model, with an R² score of **0.83** when using the top 5 most important features.
- **Gradient Boosting (GB)** also performed well, achieving an R² score of **0.73** with **porosity** being its most significant predictor.
  
All models were tested with different **train-test splits** (70/30 and 80/20) to assess their sensitivity to the size of training data.

## **Project Structure**

The project is structured as follows:

- `data/`: Contains the rock sample dataset used for training and testing the models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), feature selection, and model training.
- `models/`: Pre-trained machine learning models ready for use or further tuning.
- `results/`: Visualizations and metrics evaluating model performance, including prediction errors, R² scores, and feature importance graphs.
- `README.md`: This documentation file outlining the project's purpose, structure, and results.

## **Installation & Requirements**

This project requires **Python 3.8+** and the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `jupyter`
- `Avizo` (for pore-scale modeling and data extraction)

You can install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## **How to Run the Project**

1. Clone the repository to your local machine.
2. Place the dataset in the `data/` folder.
3. Open and run the Jupyter notebooks in the `notebooks/` folder to explore the data, select features, and train the machine learning models.
4. Pre-trained models are saved in the `models/` directory and can be loaded to make predictions on new data.
5. Review the results in the `results/` folder for detailed model performance metrics and insights.

## **Key Results**

- **Random Forest** achieved the highest accuracy with an R² score of **0.83**, demonstrating its suitability for permeability prediction in heterogeneous rock samples.
- Models such as **Gradient Boosting** and **Lasso Regression** provided useful insights into feature importance, with **porosity** and **pore radius** being critical predictors across different models.
- **Feature sensitivity analysis** showed that using 5 features optimized model performance, while increasing the feature set beyond this did not significantly improve accuracy.

## **Contributors**

- **Bakytzhan K. Assilbekov**, Satbayev University
- **Nurlykhan Ye. Kalzhanov**, KBTU BIGSoft
- **Darezhat A. Bolysbek**, Kazakh National University
- **Kenboy Sh. Uzbekaliyev**, Satbayev University

## **Acknowledgments**

This research was supported by the **Committee of Science of the Ministry of Science and Higher Education of Kazakhstan** under the project **BR18574136**: *"Development of Deep Learning and Intelligent Analysis Methods for Solving Complex Problems in Mechanics and Robotics."*

## **License**

This project is licensed under the MIT License. See the `LICENSE.md` file for details.

---


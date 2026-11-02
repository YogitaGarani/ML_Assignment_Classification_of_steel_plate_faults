# Steel Plates Faults and Metallurgical Defect Detection System

## a. Problem Statement
This project uses machine learning pipelines to detect and classify surface defects on hot-rolled steel plates. Automated anomaly detection is crucial for manufacturing quality control, minimizing raw material waste, reducing processing downtime, and ensuring structural safety in industrial supply chains.
The major challenge with this dataset involves a highly imbalanced multiclass classification task: predicting **7 distinct fault types** ('Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', and 'Other_Faults') using **27 geometric, tracking, topological, and luminescent structural metrics** extracted from digital images of plate surfaces.

## b. Dataset Description
The model pipeline leverages the **UCI Steel Plates Faults Dataset**.
* **Instance Count:** 1,941 samples
* **Feature Attributes:** 27 structural continuous variables (e.g., coordinates `X_Minimum`, surface parameter `Pixels_Areas`, luminous metrics `Luminosity_Index`, etc.)
* **Targets Evaluated:** Multi-class classification target synthesized out of 7 categories: `Pastry`, `Z_Scratch`, `K_Scratch`, `Stains`, `Dirtiness`, `Bumps`, and `Other_Faults`.

## c. GitHub Repository Link
[Github repository link](https://github.com/YogitaGarani/ML_Assignment_Classification_of_steel_plate_faults)

## d. Models Used & Evaluation Comparison
The evaluation metrics below were calculated on the training data vector set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7118 | 0.9269 | 0.6901 | 0.7024 | 0.6929 | 0.6328 |
| Decision Tree | 0.6913 | 0.8997 | 0.6369 | 0.6324 | 0.6290 | 0.6021 |
| KNN | 0.7376 | 0.9197 | 0.7383 | 0.7755 | 0.7538 | 0.6707 |
| Naive Bayes | 0.6278 | 0.9135 | 0.6573 | 0.7199 | 0.6539 | 0.5704 |
| Support Vector Machine | 0.7530 | 0.9453 | 0.7661 | 0.7381 | 0.7484 | 0.6818 |

### Observations About Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Yields a balanced F1-score (0.6929) and high AUC (0.9269), reflecting stable global boundary definitions. However, its confusion matrix reveals specific errors, frequently misclassifying true rare classes as the majority `Other_Faults` class (Class 6). |
| **Decision Tree** | Captures non-linear edge patterns relatively well, resulting in an MCC score of 0.6021. However, it displays a tendency to overfit deep leaf branches on noisy parameters like pixel boundaries, causing it to clip minority classes like `Dirtiness` down to a lower true capture rate. |
| **K-Nearest Neighbors (KNN)** | Demonstrates strong performance with a **Recall score of 0.7755** and an F1-score of 0.7538. By relying on spatial neighborhood boundaries, it accurately isolates minority clusters, making it the most sensitive model for identifying rare structural faults. |
| **Naive Bayes** | Struggles relative to the other pipelines, registering the lowest global Accuracy (0.6278) and lowest MCC (0.5704). This drop occurs because the assumption of independent conditional distribution breaks down under highly correlated spatial dimensions like bounding coordinates. |
| **Support Vector Machine (SVM)** | Achieves the highest baseline **Accuracy (0.7530)**, best overall **Precision (0.7661)**, and most reliable discrimination envelope (**0.9453 AUC**). It forms clean decision margins that successfully buffer against class imbalance noise. |
| **Overall Winner** | **Support Vector Machine (SVM)** emerges as the overall winner. It achieves top scores across Accuracy, AUC, Precision, and MCC. While KNN offers a slightly higher Recall, SVM provides the most balanced and statistically robust decision surface for production deployment. |

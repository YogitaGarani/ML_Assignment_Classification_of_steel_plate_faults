# Steel Plates Faults Classification & Deployment (Assignment 2)

## a. Problem Statement
The objective of this project is to implement an automated predictive modeling pipeline to identify seven distinct types of steel plate manufacturing faults based on 27 geometric and geometric-luminosity parameters. Early detection of defects prevents processing flaws, optimizes product distribution, and establishes reliable quality control.

## b. Dataset Description
The model pipeline leverages the **UCI Steel Plates Faults Dataset**.
* **Instance Count:** 1,941 samples
* **Feature Attributes:** 27 structural continuous variables (e.g., coordinates `X_Minimum`, surface parameter `Pixels_Areas`, luminous metrics `Luminosity_Index`, etc.)
* **Targets Evaluated:** Multi-class classification target synthesized out of 7 categories: `Pastry`, `Z_Scratch`, `K_Scratch`, `Stains`, `Dirtiness`, `Bumps`, and `Other_Faults`.

## c. GitHub Repository Link
[Insert Your Repository Live Hyperlink Here]

## d. Models Used & Evaluation Comparison
The evaluation metrics below were calculated by loading the test data vector splits inside the workspace environment:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7410 | 0.9234 | 0.6954 | 0.6120 | 0.6382 | 0.6721 |
| Decision Tree | 0.7256 | 0.8210 | 0.6841 | 0.6902 | 0.6865 | 0.6624 |
| KNN | 0.7513 | 0.9124 | 0.7230 | 0.6315 | 0.6621 | 0.6854 |
| Naive Bayes | 0.6106 | 0.8842 | 0.5821 | 0.6612 | 0.5720 | 0.5432 |
| Support Vector Machine | 0.7839 | 0.9412 | 0.7912 | 0.6894 | 0.7241 | 0.7289 |

### Observations About Model Performance
* **Logistic Regression:** Shows highly stable learning curves and excellent generalization metrics. Serves as a great baseline but minor underfitting is present due to complex decision bounds.
* **Decision Tree:** Demonstrates high execution efficiency but displays slightly lower generalized AUC scores compared to ensemble boundaries due to typical splitting variance.
* **KNN:** Strong grouping ability based on metric distances, but suffers slight latency inflation scaling up against feature counts.
* **Naive Bayes:** Shows lower overall structural accuracy because it assumes complete independence among all 27 highly correlated spatial geometric variables.
* **Support Vector Machine:** **Overall Winner for this Dataset**. Captures non-linear dependencies across structural measurements using RBF kernel features, yielding the top F1, AUC, and MCC scores.


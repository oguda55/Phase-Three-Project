# SyriaTel Clients Churn Analysis
## Project Overview

SyriaTel Communications, a leading telecommunications provider, is facing challenges related to customer churn—the phenomenon where customers discontinue their services. Churn not only results in direct revenue loss but also incurs additional costs related to acquiring new customers.

To address this, the project aims to build a machine learning model capable of predicting customer churn. This involves performing Exploratory Data Analysis (EDA) to uncover key behavioral patterns and using classification algorithms such as Logistic Regression and Decision Trees to identify customers at risk of leaving. By proactively identifying potential churners, SyriaTel can implement targeted retention strategies such as personalized offers or improved customer service, ultimately improving customer loyalty and reducing churn rates.

This project demonstrates how data-driven insights can enable strategic decision-making, helping SyriaTel optimize customer retention efforts and enhance long-term profitability.

## 1 **Business Understanding**
 The primary objective is to develop a machine learning model that accurately predicts customer churn. This predictive capability will enable SyriaTel to:

- Recognize patterns leading to churn.

- Implement targeted retention strategies.

- Ultimately reduce customer turnover and associated costs.

#### **Problem Statement**
SyriaTel lacks a predictive mechanism to identify which customers are likely to leave the service. As a result, retention efforts are reactive rather than proactive, leading to higher operational costs and loss of revenue. The problem at hand is to:

Build a classification model that can predict whether a customer is likely to churn, based on their behavioral and account-related data.

####  **Objective**
- To determine distribution of the churn 
- To establish correlation matrix for numerical features
- To determine relationship of churn in relation to important features
- To predict whether a customer will "soon" stop doing business with SyriaTel telecommunications company.  
- To find a way to prevent customer churn.
- To find a way of reducing revenue loss incase of customer churn.

#### **Challanges**
- Data Quality & Imbalance: Churn datasets often contain imbalanced classes, where non-churners vastly outnumber churners.

- Feature Relevance: Determining which customer attributes (e.g., usage patterns, service interactions) are most predictive of churn.

- Interpretability: Business stakeholders require not just predictions, but understandable insights to act upon.

- Cost Sensitivity: Misclassifying a churner as a non-churner has higher financial consequences than the reverse.

####  **Proposed Solution**
- Exploratory Data Analysis (EDA): Understand key trends and correlations in customer data.

- Predictive Modeling: Use Logistic Regression and Decision Tree Classifiers to model churn risk.

- Feature Engineering: Identify and transform relevant features such as call durations, customer service interactions, and usage patterns.

- Model Evaluation: Emphasize recall (true positive rate) to minimize false negatives—critical for catching churners early.

- Actionable Insights: Derive key drivers of churn to support targeted interventions.


####  **Conclusion**
- The successful implementation of a churn prediction model can provide SyriaTel with actionable insights into customer behavior, empowering the company to intervene before customers leave. This data-driven approach not only improves retention but also enhances customer satisfaction and reduces operational costs related to acquiring new clients. Through a blend of analytics and strategic action, SyriaTel can transform churn management into a competitive advantage.

## 2. **Data Understanding**
- Here we explore the dataset to understand its structure, content, and quality to assess its suitability for predicting customer churn for SyriaTel.

####  **Data Source:**

- The dataset is sourced from: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset/data.

 **Target Variable**: `churn` which is `True` for customers who left, `False` for those who stayed.
  
- **Features**: The dataset includes a mix of numerical and categorical features related to customer demographics, account details, usage patterns, and interactions with customer service.

#### **Exploratory Data Analysis**

**distribution of the churn variable**
![![Distribution of churn.png](<Distribution of churn.png>)]
Interpretation: The plot typically shows a much higher bar for non-churned customers (0) compared to churned customers (1), indicating that most customers stayed with the company while a smaller proportion left.
Insight: his highlights class imbalance in the dataset, which is common in churn problems. It means special care is needed during modeling to ensure the minority class (churners) is not overlooked.

**correlation matrix for numerical features**
![![correlation heatmap.png](<correlation heatmap.png>)]
High positive values (close to 1) indicate a strong direct relationship between two features (as one increases, so does the other).
High negative values (close to -1) indicate a strong inverse relationship (as one increases, the other decreases).
Values near 0 mean little or no linear relationship

**relationship between customer service calls and churn**
![![relationship between customer service calls and churn.png](<relationship between customer service calls and churn.png>)]

## 3.  **Modeling**

- We build and train on two classifiers, **Logistic Regression** and **Decision Tree**, to predict customer churn for SyriaTel.
  
- The goal is to create models that accurately identify customers at risk of churning, enabling targeted retention strategies.

#### **Feature Importance**

Feature importance analysis was performed in the Modeling phases to identify which features such as customer service calls, total day minute that most strongly influence the model classifier’s predictions.

#### **Logistic Regression Feature Importance**
[![logistic regression key feature.png](<logistic regression key feature.png>)]
 From the above, we clearly see that features such as customer service calls, international plans and total minutes, strongly influence if a customer will churn or not. This provides SyriaTel with key areas to focus on to reduce the churn percentages.

 [![Decision Tree key Feature-1.png](<Decision Tree key Feature-1.png>)]

 Feature importance was calculated for the Decision Tree which assigns a score to each feature based on its contribution to reducing impurity in the tree’s splits. From the results we observe:
- `total day minutes`: is the top predictor, reflecting inefficiency.
- `Customer service calls`: also records high number of churn showing dissatisfaction
- `international plan`: Suggests specific plan-related churn risks.

#### **Benefits for SyriaTel**

1. **Targeted Retention Efforts**:
   - Feature importance enables SyriaTel to focus on the 103 predicted churners from decision tree (73 true positives + 50 false positives) by prioritizing those with high values in key features. This will help them in resource allocation and minimizes costs.

2. **Cost Efficiency**:
   - By identifying the most impactful features, SyriaTel avoids wasting resources on irrelevant factors.

3. **Strategic Decision-Making**:
   - Insights from feature inform broader strategies, such as improving customer service processes or adjusting pricing for high-usage customers.

## 4.  **Model Evaluation**

- Here we evaluate how our models have performed. The major focus is the `Recal` as the main goal of SyriaTel is to predict churn.
- We use the confusion matrix to analyze how good our model actually predicts churn.

#### **Logistic Regression Confusion matrix**
[![logistic regression confusion matrix.png](<logistic regression confusion matrix.png>)]
True Negative (TN)= 435:** The number of customers who did not churn and were correctly predicted as non-churners.

-**False Positive (FP)= 135:** The number of customers who did not churn but were incorrectly predicted as churners (Type I error).

-**False Negative (FN)=26:** The number of customers who churned but were incorrectly predicted as non-churners (Type II error).

-**True Positive (TP)=71:** The number of customers who churned and were correctly predicted as churners.

**In summary;**
- High TN and TP mean this model is making correct predictions.
- High FP means the model is flagging too many loyal customers as churners, which could waste retention resources.
- High FN means the model is missing actual churners, which could lead to revenue loss.
#### **Strengths**
**Recall (73.20%):**
The model correctly identifies 73.20% of actual churners (71 out of 97). This is a key strength because, for SyriaTel, catching as many customers at risk of leaving as possible is critical. High recall means the model is effective at flagging most customers who are likely to churn, allowing the business to take proactive retention actions.

**Accuracy (75.86%):**
The model correctly predicts the outcome for about 76% of all customers. This shows the model performs well overall. However, because the dataset is imbalanced having more non-churners than churners, accuracy alone can be misleading. Still, a high accuracy indicates that the model is making correct predictions for the majority of cases


#### **Weaknesses**
-**Precision (34.63%):** Only about one-third of customers predicted to churn actually do. This means that many customers who are flagged as likely to leave are actually loyal and would have stayed. As a result, SyriaTel may waste resources (such as special offers or retention campaigns) on customers who are not at risk.

-**High False Positive Rate (134 customers):** The model incorrectly predicts that 134 non-churners will churn. This increases operational costs because the company might target too many customers with unnecessary retention efforts.

-**F1-Score (47.02%):** The F1-score is a balance between precision and recall. A moderate F1-score indicates that the model does not perform exceptionally well in both catching churners and avoiding false alarms. This means there is a trade-off: while the model catches most churners (high recall), it also mislabels many loyal customers as churners (low precision).

#### Decision Tree Model Evaluation
[![Decision Tree Confusion matrix.png](<Decision Tree Confusion matrix.png>)]
#### **Summary of Decision Tree Prediction**

**-Accuracy: 88.90%** — The model correctly predicts 88.90% of all customer cases.

**-Recall: 75.26%**— It successfully identifies 75.26% of actual churners (73 out of 97), which is crucial for targeting at-risk customers.

**-Precision: 59.35%** — Of all customers predicted to churn, 59.35% actually do, reducing wasted retention efforts compared to Logistic Regression.

**-F1-Score: 66.36%** — Shows a good balance between precision and recall.

**-ROC-AUC: 83.24%**— Indicates strong ability to distinguish between churners and non-churners.

#### **Confusion Matrix:**
**True Positives (TP):** 73 — Correctly predicted churners.

**True Negatives (TN):** 520 — Correctly predicted non-churners.

**False Positives (FP):** 50 — Non-churners incorrectly flagged as churners.

**False Negatives (FN):** 24 — Churners missed by the model.

#### **Business Implications:**
- The Decision Tree model captures most churners, enabling proactive retention.
- It reduces false positives compared to Logistic Regression, optimizing resource allocation.
Some churners are still missed, but overall, the model is efficient and reliable for deployment.

#### **Comparison with Logistic Regression**
#### **Precision:**
**Decision Tree:** 59.35%
**Logistic Regression:** 34.46%.
The Decision Tree is much better at ensuring that customers flagged as churners are actually at risk, reducing wasted retention efforts
#### **Recall:**
**Decision Tree:** 75.26%
**Logistic Regression:** 73.20%.
The Decision Tree is still better than logistic Regression while measuring the proportion of actual churners correctly identified by the model.
#### **Accuracy:**
**Decision Tree:** 88.90%
**Logistic Regression:** 75.86%.
The Decision Tree makes more correct predictions overall, especially for non-churners.
#### **False Positives:**
**Decision Tree:** 50
**Logistic Regression:** 135.
The Decision Tree flags far fewer loyal customers as churners, saving resources.
#### **F1-Score:**
**Decision Tree:** 66.36 %
**Logistic Regression:** 47.02%.
The Decision Tree achieves a better balance between precision and recall.
#### **ROC-AUC:**
**Decision Tree:** 83.24%
**Logistic Regression:** 74.76%
The Decision Tree is better at distinguishing churners from non-churners

## 5.  **Recommendations for SyriaTel**
**Deploy the Decision Tree Model:**
Use the Decision Tree classifier in your CRM system to identify customers at high risk of churning. This model offers higher precision and accuracy, reducing wasted retention efforts.

**Target At-Risk Customers:**
Focus retention campaigns such as special offers, discounts, or improved service on the customers flagged as likely to churn. This will help reduce revenue loss and improve customer loyalty.

**Monitor and Adjust:**
Regularly monitor the model’s performance and update it with new data to maintain accuracy. Analyze the cases where churners are missed (false negatives) and adjust the model or intervention strategies as needed.

**Address Key Churn Drivers:**
Pay special attention to customers with frequent customer service calls, as this is a strong churn indicator.
Consider reviewing and improving international plan features and voice mail plans, as these are also important predictors.

**Optimize Resource Allocation:**
By reducing false positives, the Decision Tree model helps SyriaTel allocate retention resources more efficiently, focusing on customers who are truly at risk.

**Continuous Improvement:**
Investigate the reasons behind customer churn and use insights from the model to inform business decisions, product improvements, and customer service enhancements.

#### **Conclusion:**
The Decision Tree outperforms Logistic Regression in almost every metric, especially in precision and overall accuracy. It is the preferred model for deployment, as it reduces unnecessary retention actions and provides more reliable predictions for SyriaTel.



  





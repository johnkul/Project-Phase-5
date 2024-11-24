# Bank Customer Complaints Classification


## Executive Summary

This project uses Natural Language Processing (NLP) to streamline customer complaint classification, reducing the time and complexity of complaint submission. By deploying an NLP model that automatically categorizes complaints based on content, the system eliminates the need for extensive questions or options, enhancing user experience with a simple interface.

We trained and evaluated various models, including Multinomial Naive Bayes, Support Vector Machine (SVM), Logistic Regression, Random Forest, and ExtraTrees, ultimately selecting the transformer-based BERT model for deployment. BERT demonstrated strong performance, achieving a Macro F1-score of 0.85, Weighted F1-score of 0.89, and an accuracy of 89%, providing balanced and reliable classification across complaint categories.

Deployed on Hugging Face with a Streamlit interface, the solution allows customers to input complaint text and contact details, which the BERT model then categorizes. Notifications are sent through Africastalking’s SMS API to both the customer and relevant support teams, ensuring immediate updates.

Future improvements recommended include implementing a feedback loop for continuous retraining, optimizing specific complaint types, and expanding notification channels, further enhancing the system’s accuracy, adaptability, and responsiveness.


## Project Overview

This project uses Natural Language Processing (NLP) to automatically classify customer complaints, minimizing the number of survey questions customers need to answer. By training an NLP model to understand complaint content, the system simplifies the submission process and enhances customer experience. The main goals are to reduce complaint logging time and to create a straightforward, user-friendly platform.

**Objectives:**
- Train an NLP model to categorize complaints into predefined classes.
- Streamline the complaint logging experience.
- Improve bank responsiveness through faster, accurate complaint classification.


**Model Training, Evaluation, and Selection:**

After thorough data exploration and preprocessing (handling missing values, duplicate removal, text standardization, TF-IDF transformation, and scaling), we trained various models suitable for text classification, including Multinomial Naive Bayes, SVM, Logistic Regression, Random Forest, and an ExtraTrees ensemble. We also tested a transformer model, BERT, which achieved the best results with a Macro F1-score of 0.85, Weighted F1-score of 0.89, and 89% accuracy, providing consistent performance across all categories. Macro F1 was prioritized for balanced classification, supported by Weighted F1 to account for class imbalance.

**Deployment and Application:**

The solution, deployed with Streamlit on Hugging Face, offers a simple interface for complaint submission. Customers enter complaint text, phone number, and account number, which the BERT model categorizes in real-time. An integrated SMS notification system (via Africastalking) sends classified complaints to the relevant support team and a copy to the customer, ensuring prompt and organized handling.

**Limitations and Future Work:**

The BERT model performs well overall, but accuracy varies by complaint type, indicating a need for further tuning. SMS delivery is currently limited to Airtel and Telkom due to restrictions on Safaricom networks, and email notifications are slow. Future enhancements include implementing a feedback loop for continuous model improvement, expanding notification channels, and optimizing for specific complaint types to improve system reliability and adaptability.


## Business Understanding

**Problem Statement:**  
Customers of financial institutions often face frustration and dissatisfaction when submitting complaints, due to complex processes involving multiple options and repetitive chatbot questions. This setup slows down complaint submission and adds to customer frustration, especially when swift resolution is needed. There is a clear need for a streamlined, efficient complaint system that minimizes unnecessary steps.

**Root Causes:**
- **Complex Navigation:** Current platforms have cumbersome menus and options.
- **Inefficient Chatbots:** Chatbots often ask redundant or irrelevant questions, increasing submission time.
- **Lack of Personalization:** One-size-fits-all pathways don’t address specific customer needs.
- **Limited Data Utilization:** Current systems don’t fully leverage data to streamline the complaint process, causing repetitive requests for information.

**Key Stakeholders:**
1. **Customers:** Seek a quick, hassle-free way to submit complaints with minimal steps, along with assurance of timely resolutions.
2. **Customer Support Teams:** Require efficient complaint categorization to handle issues quickly, access to accurate complaint data, and tools for tracking complaint status to maintain service quality.


## Data understanding

The Consumer Complaints Dataset from the Consumer Financial Protection Bureau (CFPB) offers real-world data on consumer complaints about financial products and services, making it ideal for NLP projects. This publicly accessible dataset includes detailed complaint narratives across five categories:

- Credit Reporting
- Debt Collection
- Mortgages and Loans (e.g., car, payday, student loans)
- Credit Cards
- Retail Banking (e.g., checking/savings accounts, money transfers)

The version used, sourced from Kaggle, contains approximately 162,400 records with varying narrative lengths. While categorized, the data is imbalanced, with 56% focused on credit reporting, posing a challenge for balanced model training. Tailored strategies are essential to ensure accurate classification across all complaint types.


## Data Exploration & Preparation

The initial data examination involved removing unnecessary columns and handling missing values in the narrative field. We identified 37,735 duplicate entries but retained them, as removing duplicates negatively impacted model performance. Class distribution analysis revealed a notable imbalance, especially in the credit reporting category, which was addressed through stratified splits and later through Synthetic Minority Over-sampling Technique (SMOTE). A text length analysis showed a right-skewed distribution, with most narratives under 1,000 characters, aligning with typical complaint lengths.

The data preparation process involved comprehensive preprocessing and transformation steps to ensure clean, consistent, and usable data for modeling. During **data preprocessing**, text was standardized by converting it to lowercase, removing special characters and numbers, and handling whitespace. Tokenization split the text into individual words, while stop word removal filtered out common but uninformative words. Lemmatization reduced words to their root forms, enhancing the model’s ability to recognize related terms as a single concept. This preprocessing pipeline produced a well-processed `cleaned_narrative` column, ready for numerical feature extraction.

For **data transformation**, we split the data into training and testing sets with stratified sampling to maintain class balance. The text was then vectorized using TF-IDF to represent words as weighted features, capturing term importance. Finally, MinMax scaling was applied to standardize the feature values, resulting in `X_train_scaled` and `X_test_scaled` datasets optimized for model training and evaluation.


## Model Training, Evaluaton, Improvement & Selection

We trained a series of baseline models, including Multinomial Naïve Bayes, Support Vector Machine (SVM), Logistic Regression, and Random Forest, to identify the most effective method for classifying customer complaints. Model evaluation focused on classification report metrics—accuracy, precision, recall, and F1-score for each class—with a particular emphasis on **Macro F1-score** and **Weighted F1-score**. Macro F1 provided an equal-weighted average across all classes, aligning with our goal to treat each complaint category equally, while Weighted F1 accounted for class imbalance by assigning more weight to larger classes, offering a broader perspective on model performance.

For Model Improvement, we started by tuning the Random Forest model to enhance performance through parameter adjustments. We then applied SMOTE to balance the dataset and retrained both a Random Forest and an ensemble ExtraTrees model on the resampled data, comparing their results with the tuned model. Finally, we explored a transformer-based model, **BERT**, which showed strong suitability for text classification tasks due to its deep language understanding. After evaluating performance across all metrics, we selected BERT as the most suitable model for deployment since it achieved robust accuracy and balanced performance across all complaint categories and for its appropriateness in natural language processing tasks.


## Deployment and Application

To achieve the project goal of simplifying the customer complaint process, we developed and deployed a streamlined solution prioritizing efficiency, ease of use, and fast issue categorization.

1. **User-Friendly Interface**: Built with Streamlit, this interface allows bank customers to easily enter their complaint, phone number, and account number. The system then categorizes complaints based on the model's predictions, creating a simplified submission process.

2. **Model Integration**: The pretrained BERT model is integrated with the Streamlit app to classify complaints in real time, processing customer inputs to determine the appropriate complaint category.

3. **Automated Notification System**: Integrated with Africastalking’s SMS API, the system sends categorized complaints directly to the designated support team phone numbers, and provides the customer with a copy of their complaint via SMS. SMS was chosen over email due to email latency issues.

4. **Deployment on Hugging Face**: The model and Streamlit app were deployed on Hugging Face, accessible via a link, allowing customers to submit complaints quickly without navigating complex options.

5. **Continuous Improvement**: While not implemented initially, a feedback loop is recommended to retrain the model with new complaint data, enhancing accuracy and adaptability over time.

**Challenges**:
- SMS notifications are limited to Airtel and Telkom due to Safaricom's restrictions on promotional messages, impacting accessibility.
- Email notifications were considered but replaced with SMS due to slower delivery times.

This deployment provides an efficient, user-focused platform for automatic complaint classification, reducing customer frustration and improving response times. The BERT model's strong language processing capabilities ensure accurate classifications, making this solution a responsive and effective complaint management tool.


## Conclusion and Recommendations

BERT was chosen for its suitability in NLP tasks, particularly for its ability to capture nuanced meanings and context in complaint narratives. With a weighted F1-score of 0.89 and macro F1-score of 0.85, BERT demonstrated strong accuracy and reliability across complaint categories, making it effective for automatic classification. The model is integrated with a Streamlit interface and deployed on Hugging Face, offering users a seamless experience for complaint submission and categorization.

**Insights and Impact**:  
This deployment meets project objectives by streamlining the complaint submission process. The system reduces customer friction by allowing direct input of complaints without complex options, automatically categorizing complaints and routing notifications to the appropriate support team via SMS for faster resolution. This setup improves customer experience by reducing wait times and simplifying the submission process.

**Limitations and Future Work**:  
While BERT performs well overall, its accuracy may vary across complaint types, suggesting a need for further tuning or data augmentation. Currently, SMS notifications are limited to Airtel and Telkom networks due to restrictions with Safaricom, and email was too slow. Future improvements include implementing a feedback loop for continuous retraining, expanding notification options, and optimizing model performance for specific categories. These enhancements would strengthen the solution’s reliability and responsiveness, adapting to evolving customer needs.

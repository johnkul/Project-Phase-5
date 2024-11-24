import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import africastalking
import os

# Initialize Africa's Talking with hardcoded credentials
USERNAME = "getsome"
API_KEY = "atsk_80b3dfba0d92ca2773b8fa753f230c15d6b140baf26537da41f923cf66aebc92599473c6"

# Initialize Africa's Talking
africastalking.initialize(USERNAME, API_KEY)
sms = africastalking.SMS

# Load the saved model and tokenizer
model_dir = "saved_bert_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

# Define the label mapping and phone numbers for each category
label_map = {
    0: ("credit_card", "+254733404661"),
    1: ("retail_banking", "+254733404661"),
    2: ("credit_reporting", "+254737190058"),
    3: ("mortgages_and_loans", "+254733404661"),
    4: ("debt_collection", "+254737190058")
}

# Define the classify function
def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    category, phone_number = label_map.get(predicted_label, ("unknown", None))
    return category, phone_number

# Define the function to send SMS
def send_sms(phone_number, message):
    try:
        response = sms.send(message, [phone_number])
        print(response)
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Streamlit app
st.title("Complaint Classifier")

# Input fields with placeholders
complaint = st.text_area("Enter your complaint:", placeholder="Describe your issue in detail")
phone_number = st.text_input("Enter your phone number:", placeholder="+2547XXXXXXX")
account_number = st.text_input("Enter your account number:", placeholder="123456789")

if st.button("Classify and Send SMS"):
    if complaint and phone_number and account_number:
        category, assigned_phone_number = classify_text(complaint)
        if category == "unknown":
            message = "Sorry, this is not the place for such complaints."
        else:
            message = f"Complaint: {complaint}\nPredicted Category: {category}\nAccount Number: {account_number}"
        
        # Send SMS to user's number
        send_sms(phone_number, message)
        
        # Send SMS to assigned phone number based on category
        if assigned_phone_number:
            send_sms(assigned_phone_number, message)
        
        st.write(message)
    else:
        st.write("Please enter a complaint, phone number, and account number.")

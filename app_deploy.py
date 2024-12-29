import streamlit as st
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from random import randint
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, ToPILImage
import requests
import os
import zipfile

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # Adjust dimensions based on input size
        self.dropout = nn.Dropout(0.5)
        # Output Layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # Flatten
        x = self.flatten(x)
        # Fully Connected Layer with ReLU and Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # Output Layer
        x = self.fc2(x)
        return x

# Setup user password etc.
users = {"admin": "admin", 
        "guest": "guest",
        "tianfeng": "tianfeng"}

def authenticate(username,password):
    if username in users and users[username]==password:
            return True
    return False

def logout():
    st.session_state["authentication_status"] = False
    st.sidebar.info("You have been logged out.")

# Function to load the pre-trained model
def load_trained_model():
    num_classes = 9  # Adjust to the number of classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("cnn_model_cgan.pth", map_location=device))
    model.eval()
    return model, device

# # Load test data
# @st.cache_data
# def load_test_data():
#     X_test = np.load("X_test.npy")
#     y_test = np.load("y_test.npy")
#     return X_test, y_test

def load_test_data():
    zip_path = "testdata.zip"
    extract_dir = "testdata"

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    X_test = np.load(os.path.join(extract_dir, "X_test.npy"))
    y_test = np.load(os.path.join(extract_dir, "y_test.npy"))

    return X_test, y_test

# Generate random dates for samples
@st.cache_data
def generate_dates(num_samples):
    start_date = datetime.today() - timedelta(days=100)
    return [(start_date + timedelta(days=randint(0, 100))).strftime("%Y-%m-%d") for _ in range(num_samples)]

def set_page_config():
    st.set_page_config(layout="wide")  # This sets the layout to wide
    st.markdown(
        """
        <style>
        .main {
            padding-left: 5rem;  /* Adjust the left padding */
            padding-right: 5rem; /* Adjust the right padding */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_page_config()  # Call the function


# Main app function
def main():
    # login
    st.sidebar.title("User Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False

    if login_button:
        if authenticate(username, password):
            st.session_state["authentication_status"] = True
            st.session_state["username"] = username
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid username or password")

    if st.session_state["authentication_status"]:
        st.sidebar.button("Logout", on_click=logout)

        st.markdown(
            """
            <div style="background-color:#0e1117;padding:10px;border-radius:5px;margin-bottom:20px;">
                <h4 style="text=align:center;font-size:12px; color:white">
                    Need help? Contact Us @
                    <a href="mailto:2180850@siswa.um.edu.my" style="color:lightblue;">2180850@siswa.um.edu.my</a>
                <h/4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.title("Wafer Map Classification & Trend Analysis")
        st.success("Model loaded successfully.")

        # Load model and test data
        model, device = load_trained_model()
        X_test, y_test = load_test_data()
        label_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
        dates = generate_dates(len(y_test))

        # Add dropdown for selecting a label
        selected_label = st.selectbox("Select a Label", label_names)
        label_idx = label_names.index(selected_label)

        # Filter data based on the selected label
        filtered_indices = np.where(y_test == label_idx)[0]
        filtered_X_test = X_test[filtered_indices]
        filtered_dates = np.array(dates)[filtered_indices]

        # Date Range
        st.subheader("Filter by Date Range")
        min_date = datetime.strptime(min(filtered_dates), "%Y-%m-%d")
        max_date = datetime.strptime(max(filtered_dates), "%Y-%m-%d")
        date_range = st.date_input("Select date range",value = (min_date,max_date),min_value = min_date, max_value = max_date)

        if len(date_range) ==2:
            start_date, end_date = date_range
            mask = (pd.to_datetime(filtered_dates) >= pd.Timestamp(start_date)) & (pd.to_datetime(filtered_dates) <= pd.Timestamp(end_date))
            filtered_X_test = filtered_X_test[mask]
            filtered_dates = filtered_dates[mask]

        sorted_indices = np.argsort(pd.to_datetime(filtered_dates))
        filtered_X_test = filtered_X_test[sorted_indices] # reorder maps
        filtered_dates = filtered_dates[sorted_indices] # reorder dates

        # Display wafer maps
        st.subheader(f"Wafer Maps for Label: {selected_label}")
        num_rows = len(filtered_X_test) // 10 + 1
        for i in range(num_rows):
            cols = st.columns(10)
            for j in range(10):
                idx = i * 10 + j
                if idx < len(filtered_X_test):
                    img = filtered_X_test[idx].transpose(1, 2, 0)
                    with cols[j]:
                        st.image(img, caption=f"Date: {filtered_dates[idx]}", use_container_width=True)

        # Show trends over time
        st.subheader("Trend of Wafer Maps Over Time")
        trend_df = pd.DataFrame({
            'Date': pd.to_datetime(dates, errors='coerce'),
            'Label': [label_names[int(label)] for label in y_test]
        })
        trend_count = trend_df.groupby(['Date', 'Label']).size().reset_index(name='Count')
        trend_filtered = trend_count[trend_count['Label'] == selected_label]

        # Debugging outputs
        # st.write("Filtered Trend DataFrame (Before Cleanup):", trend_filtered)

        # Data cleaning and debugging
        trend_filtered['Date'] = pd.to_datetime(trend_filtered['Date'], errors='coerce')
        trend_filtered['Count'] = pd.to_numeric(trend_filtered['Count'], errors='coerce')
        trend_filtered = trend_filtered.dropna(subset=['Date', 'Count'])
        trend_filtered['Count'] = trend_filtered['Count'].astype(int)

        # Generate a complete date range for the selected date range
        date_range_full = pd.date_range(start=start_date, end=end_date)
        default_trend_df = pd.DataFrame({'Date': date_range_full, 'Label': selected_label, 'Count': 0})
        trend_filtered = pd.merge(default_trend_df, trend_filtered, on=['Date', 'Label'], how='left')
        trend_filtered['Count'] = trend_filtered['Count_y'].fillna(0).astype(int)
        trend_filtered = trend_filtered[['Date', 'Label', 'Count']]

        # Ensure the trend_filtered DataFrame is sorted by Date
        trend_filtered = trend_filtered.sort_values(by="Date")

        # Debugging output
        st.write("Filtered Trend DataFrame:")
        st.write(trend_filtered)
    
        # Ensure the DataFrame is not empty before plotting
        if trend_filtered.empty:
            st.warning(f"No valid data to plot for label: {selected_label} within the selected date range.")
        else:
            plt.figure(figsize=(20, 8))
            plt.plot(trend_filtered['Date'], trend_filtered['Count'], marker='o')
            plt.xticks(rotation=45)
            plt.title(f"Trend for Label: {selected_label}")
            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.grid()
            plt.tight_layout()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
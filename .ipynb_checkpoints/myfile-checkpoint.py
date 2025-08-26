import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the trained model
with open("project.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Handwritten Digit Recognition")
st.write("This app uses a Logistic Regression model trained on the scikit-learn digits dataset.")

# Allow user to draw the digit (8x8) using sliders
st.subheader("Draw your digit (8x8 grid with values 0 to 16):")

input_grid = []
for i in range(8):
    cols = st.columns(8)
    row = []
    for j in range(8):
        pixel = cols[j].number_input(f"{i},{j}", min_value=0, max_value=16, value=0, key=f"{i}-{j}")
        row.append(pixel)
    input_grid.extend(row)

if st.button("Predict"):
    input_data = np.array(input_grid).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Digit: {prediction}")

    # Display the digit
    st.subheader("You entered:")
    fig, ax = plt.subplots()
    ax.matshow(np.array(input_grid).reshape(8, 8), cmap='gray')
    st.pyplot(fig)

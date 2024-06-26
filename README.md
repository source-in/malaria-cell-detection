### README for Malaria Cell Detection Web Application

#### Objective
This Streamlit web application allows users to upload an image of a cell to predict if it is infected with malaria using a custom ResNet50 model.

#### Installation and Setup

1. **Clone the Repository**
   ```sh
   git clone https://github.com/source-in/malaria-cell-detection.git
   cd malaria-cell-detection
   ```

2. **Set Up Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Required Libraries**
   ```sh
   pip install -r requirements.txt
   ```

4. **Ensure the Model File is Present**
   - Make sure `malaria_detection_model.h5` is in the same directory as `app.py`.

#### Running the Application

1. **Start the Streamlit App**
   ```sh
   streamlit run app.py
   ```

2. **Usage**
   - Open your browser and go to `http://localhost:8501`.
   - Upload an image of a cell to get a prediction (Parasitized or Uninfected).

#### Libraries Used
- Streamlit
- TensorFlow
- Pillow
- NumPy

---

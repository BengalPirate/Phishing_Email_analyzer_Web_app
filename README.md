# Phishing Email NLP Analysis - Streamlit App

A comprehensive Streamlit web application for analyzing phishing emails using Natural Language Processing (NLP) techniques with NLTK.

## Features

- **Dataset Overview**: Interactive exploration of the phishing email dataset with statistics and visualizations
- **Text Preprocessing**: Step-by-step demonstration of text cleaning, tokenization, stopword removal, stemming, and lemmatization
- **Word Cloud Visualization**: Visual representation of word frequencies in phishing vs. legitimate emails
- **NLTK Analysis**: Advanced NLP analysis including POS tagging, Named Entity Recognition, n-grams, and vocabulary richness

## Prerequisites

- Python 3.9 or higher
- Phishing Email Dataset from Kaggle (see [Dataset Setup](#dataset-setup) section below)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application Locally

1. Navigate to the project directory:
```bash
cd /path/to/AI_Detection_5
```

2. Run the Streamlit app:
```bash
streamlit run phishing_nlp_app.py
```

3. The app will open in your default web browser at `http://localhost:8501`

## Project Structure

```
AI_Detection_5/
├── phishing_nlp_app.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore file
└── data/
    └── phishing_email.csv   # Dataset file (download from Kaggle - see Dataset Setup)
```

## Cloud Deployment Options

### 1. Streamlit Community Cloud (Recommended - Free)

1. Push your code to a GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Set the main file path to `phishing_nlp_app.py`
6. Click "Deploy"

### 2. Heroku

1. Install the Heroku CLI: [https://devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

2. Create a `setup.sh` file:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Create a `Procfile`:
```
web: sh setup.sh && streamlit run phishing_nlp_app.py
```

4. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### 3. AWS EC2

1. Launch an EC2 instance (Amazon Linux 2 or Ubuntu)
2. SSH into the instance
3. Install Python and pip
4. Clone your repository
5. Install dependencies: `pip install -r requirements.txt`
6. Run the app: `streamlit run phishing_nlp_app.py --server.port 8501 --server.address 0.0.0.0`
7. Configure security groups to allow traffic on port 8501

### 4. Google Cloud Platform (Cloud Run)

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD streamlit run phishing_nlp_app.py --server.port 8501 --server.address 0.0.0.0
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy phishing-nlp-app --source . --platform managed --region us-central1 --allow-unauthenticated
```

### 5. Azure Web Apps

1. Install Azure CLI: [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. Create a web app:
```bash
az webapp up --name your-app-name --runtime "PYTHON:3.13"
```

## Dataset Setup

### Downloading the Dataset

This application uses the **Phishing Email Dataset** from Kaggle. Follow these steps to download it:

1. Visit the Kaggle dataset page: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

2. Download the dataset (you may need to create a free Kaggle account)

3. Extract the downloaded ZIP file

4. Create a `data/` directory in the project root if it doesn't exist:
   ```bash
   mkdir -p data
   ```

5. Copy the `phishing_email.csv` file to the `data/` directory:
   ```bash
   cp /path/to/downloaded/phishing_email.csv data/
   ```

### Dataset Structure

The dataset contains two columns:
- `text_combined`: The email text content
- `label`: 0 for legitimate emails, 1 for phishing emails

### Dataset Citation

```
Alam, N. A. (2024). Phishing Email Dataset. Kaggle.
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
```

**Note**: The dataset file is approximately 102MB and contains 82,486 email samples. It is not included in this repository due to its size and licensing considerations.

## Usage Tips

1. **Sample Size**: For faster performance during testing, use the sample size slider to load fewer rows (100-1000 recommended for quick exploration)
2. **Full Analysis**: For comprehensive analysis, increase the sample size to 10,000+ rows
3. **Word Clouds**: Use the filter options to compare word frequencies between phishing and legitimate emails
4. **NLTK Downloads**: The first time you run the app, it will automatically download required NLTK data packages

## Troubleshooting

**Issue**: NLTK data not found
- **Solution**: The app automatically downloads required NLTK data on first run. If you encounter issues, manually download:
```python
import nltk
nltk.download('all')
```

**Issue**: Dataset file not found
- **Solution**: Check the dataset path in `phishing_nlp_app.py` and update it to match your file location

**Issue**: Memory errors with large datasets
- **Solution**: Reduce the sample size using the slider in the app interface

## Assignment Requirements

This application fulfills the following requirements:
- Dataset exploration and display
- Text preprocessing demonstration
- Word cloud visualization
- NLTK-based NLP analysis (POS tagging, NER, n-grams)
- Interactive web interface using Streamlit
- Cloud deployment ready

## Technologies Used

- **Streamlit**: Web application framework
- **NLTK**: Natural Language Processing
- **Pandas**: Data manipulation
- **WordCloud**: Word frequency visualization
- **Matplotlib & Seaborn**: Data visualization
- **Python 3.13**: Programming language

## License

This project is created for educational purposes as part of an AI Detection course assignment.

import streamlit as st
import pandas as pd
import numpy as np
import re
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Bypass SSL verification for NLTK downloads (handles certificate issues)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download all required NLTK resources with error handling for different versions"""
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker_tab',
        'words',
        'wordnet',
        'omw-1.4'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            # Silently handle download failures - resources may already be installed
            pass

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Phishing Email NLP Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📧 Phishing Email NLP Analysis Dashboard</h1>', unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data(sample_size=None):
    """Load the phishing email dataset"""
    df = pd.read_csv('data/phishing_email.csv')
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    return df

# Text preprocessing functions
def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stopwords from tokens"""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    """Apply stemming to tokens"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    """Apply lemmatization to tokens"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def full_preprocess(text):
    """Complete preprocessing pipeline"""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(tokens_no_stop)
    return ' '.join(lemmatized)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Dataset size selection
    st.subheader("Dataset Options")
    use_sample = st.checkbox("Use sample dataset (faster loading)", value=True)
    if use_sample:
        sample_size = st.slider("Sample size", min_value=100, max_value=10000, value=1000, step=100)
    else:
        sample_size = None
        st.warning("⚠️ Loading full dataset may take time")

    # Load data button
    if st.button("🔄 Load/Reload Data"):
        st.cache_data.clear()

    st.markdown("---")

    # Navigation
    st.subheader("📑 Navigation")
    page = st.radio(
        "Select Analysis:",
        ["Dataset Overview", "Text Preprocessing", "Word Cloud Visualization", "NLTK Analysis"]
    )

# Load data
with st.spinner("Loading dataset..."):
    df = load_data(sample_size)

st.success(f"✅ Loaded {len(df):,} email samples")

# Main content based on selected page
if page == "Dataset Overview":
    st.markdown('<h2 class="sub-header">📊 Dataset Exploration</h2>', unsafe_allow_html=True)

    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emails", f"{len(df):,}")
    with col2:
        phishing_count = df['label'].sum()
        st.metric("Phishing Emails", f"{phishing_count:,}")
    with col3:
        legitimate_count = len(df) - phishing_count
        st.metric("Legitimate Emails", f"{legitimate_count:,}")
    with col4:
        phishing_ratio = (phishing_count / len(df) * 100)
        st.metric("Phishing Ratio", f"{phishing_ratio:.1f}%")

    # Distribution visualization
    st.subheader("📈 Email Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    labels_count = df['label'].value_counts()
    ax1.bar(['Legitimate (0)', 'Phishing (1)'], [labels_count.get(0, 0), labels_count.get(1, 0)],
            color=['#2ecc71', '#e74c3c'])
    ax1.set_ylabel('Count')
    ax1.set_title('Email Distribution')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    ax2.pie([legitimate_count, phishing_count],
            labels=['Legitimate', 'Phishing'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Email Distribution')

    st.pyplot(fig)
    plt.close()

    # Sample data display
    st.subheader("📧 Sample Emails")

    filter_type = st.selectbox("Filter by:", ["All", "Legitimate Only", "Phishing Only"])

    if filter_type == "Legitimate Only":
        display_df = df[df['label'] == 0]
    elif filter_type == "Phishing Only":
        display_df = df[df['label'] == 1]
    else:
        display_df = df

    # Display settings
    num_display = st.slider("Number of rows to display:", 5, 50, 10)

    # Show dataframe
    st.dataframe(
        display_df.head(num_display)[['text_combined', 'label']].rename(
            columns={'text_combined': 'Email Text', 'label': 'Label (0=Legitimate, 1=Phishing)'}
        ),
        use_container_width=True
    )

    # Text length analysis
    st.subheader("📏 Text Length Analysis")
    df['text_length'] = df['text_combined'].str.len()

    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column='text_length', by='label', ax=ax)
    ax.set_xlabel('Label (0=Legitimate, 1=Phishing)')
    ax.set_ylabel('Text Length (characters)')
    ax.set_title('Text Length Distribution by Email Type')
    plt.suptitle('')
    st.pyplot(fig)
    plt.close()

elif page == "Text Preprocessing":
    st.markdown('<h2 class="sub-header">🔧 Text Preprocessing Pipeline</h2>', unsafe_allow_html=True)

    st.markdown("""
    This section demonstrates the text preprocessing steps applied to phishing email analysis.
    Each step transforms the raw text to prepare it for NLP analysis.
    """)

    # Select sample email
    email_index = st.selectbox("Select an email to preprocess:", range(min(20, len(df))))
    sample_text = df.iloc[email_index]['text_combined']
    sample_label = df.iloc[email_index]['label']

    st.info(f"**Email Type:** {'🚨 Phishing' if sample_label == 1 else '✅ Legitimate'}")

    # Original text
    st.subheader("📄 Step 1: Original Text")
    st.text_area("Original Email:", sample_text[:500] + "..." if len(sample_text) > 500 else sample_text, height=150)

    # Cleaned text
    st.subheader("🧹 Step 2: Cleaned Text")
    cleaned = clean_text(sample_text)
    st.text_area("Cleaned (lowercase, removed special chars):", cleaned[:500] + "..." if len(cleaned) > 500 else cleaned, height=150)

    # Tokenization
    st.subheader("✂️ Step 3: Tokenization")
    tokens = tokenize_text(cleaned)
    st.write(f"**Number of tokens:** {len(tokens)}")
    st.write("**First 50 tokens:**")
    st.write(tokens[:50])

    # Remove stopwords
    st.subheader("🚫 Step 4: Remove Stopwords")
    tokens_no_stop = remove_stopwords(tokens)
    st.write(f"**Tokens after removing stopwords:** {len(tokens_no_stop)} (removed {len(tokens) - len(tokens_no_stop)} words)")
    st.write("**First 50 tokens:**")
    st.write(tokens_no_stop[:50])

    # Lemmatization comparison
    st.subheader("📝 Step 5: Stemming vs Lemmatization")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Stemming (Porter Stemmer):**")
        stemmed = stem_tokens(tokens_no_stop[:20])
        comparison_df = pd.DataFrame({
            'Original': tokens_no_stop[:20],
            'Stemmed': stemmed
        })
        st.dataframe(comparison_df, use_container_width=True)

    with col2:
        st.write("**Lemmatization (WordNet):**")
        lemmatized = lemmatize_tokens(tokens_no_stop[:20])
        comparison_df = pd.DataFrame({
            'Original': tokens_no_stop[:20],
            'Lemmatized': lemmatized
        })
        st.dataframe(comparison_df, use_container_width=True)

    # Final preprocessed text
    st.subheader("✨ Step 6: Final Preprocessed Text")
    final_text = full_preprocess(sample_text)
    st.text_area("Final preprocessed text:", final_text, height=150)

    # Summary statistics
    st.subheader("📊 Preprocessing Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Length", len(sample_text))
    with col2:
        st.metric("After Cleaning", len(cleaned))
    with col3:
        st.metric("Token Count", len(tokens_no_stop))
    with col4:
        reduction = (1 - len(final_text) / len(sample_text)) * 100
        st.metric("Size Reduction", f"{reduction:.1f}%")

elif page == "Word Cloud Visualization":
    st.markdown('<h2 class="sub-header">☁️ Word Cloud Visualization</h2>', unsafe_allow_html=True)

    st.markdown("""
    Word clouds visualize the most frequent words in the emails. Larger words appear more frequently in the dataset.
    """)

    # Word cloud settings
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("⚙️ Settings")
        email_type = st.radio("Email Type:", ["All Emails", "Legitimate Only", "Phishing Only"])
        max_words = st.slider("Maximum words:", 50, 300, 100)
        apply_preprocessing = st.checkbox("Apply full preprocessing", value=True)

    with col2:
        # Filter data based on selection
        if email_type == "Legitimate Only":
            filtered_df = df[df['label'] == 0]
            color_scheme = 'Greens'
            title_suffix = "(Legitimate Emails)"
        elif email_type == "Phishing Only":
            filtered_df = df[df['label'] == 1]
            color_scheme = 'Reds'
            title_suffix = "(Phishing Emails)"
        else:
            filtered_df = df
            color_scheme = 'viridis'
            title_suffix = "(All Emails)"

        # Combine all text
        with st.spinner("Generating word cloud..."):
            if apply_preprocessing:
                all_text = ' '.join(filtered_df['text_combined'].apply(full_preprocess))
            else:
                all_text = ' '.join(filtered_df['text_combined'].apply(clean_text))

            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                max_words=max_words,
                background_color='white',
                colormap=color_scheme,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(all_text)

            # Display word cloud
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud {title_suffix}', fontsize=20, pad=20)
            st.pyplot(fig)
            plt.close()

    # Top words analysis
    st.subheader("📊 Top Word Frequencies")

    # Count words
    if apply_preprocessing:
        words = [word for text in filtered_df['text_combined'] for word in full_preprocess(text).split()]
    else:
        words = [word for text in filtered_df['text_combined'] for word in clean_text(text).split()
                if word not in stopwords.words('english')]

    word_freq = Counter(words)
    top_words = word_freq.most_common(20)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    words_list = [word[0] for word in top_words]
    counts_list = [word[1] for word in top_words]

    bars = ax.barh(words_list, counts_list)

    # Color bars based on email type
    if email_type == "Legitimate Only":
        for bar in bars:
            bar.set_color('#2ecc71')
    elif email_type == "Phishing Only":
        for bar in bars:
            bar.set_color('#e74c3c')
    else:
        for bar in bars:
            bar.set_color('#3498db')

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top 20 Most Frequent Words {title_suffix}', fontsize=14, pad=15)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Display table
    st.subheader("📋 Word Frequency Table")
    freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    freq_df.index = freq_df.index + 1
    st.dataframe(freq_df, use_container_width=True)

elif page == "NLTK Analysis":
    st.markdown('<h2 class="sub-header">🔬 NLTK-Based NLP Analysis</h2>', unsafe_allow_html=True)

    st.markdown("""
    This section demonstrates advanced NLP techniques using NLTK including
    Part-of-Speech tagging, Named Entity Recognition, and linguistic pattern analysis.
    """)

    # Select sample for analysis
    analysis_email_index = st.selectbox("Select an email for detailed analysis:", range(min(20, len(df))))
    analysis_text = df.iloc[analysis_email_index]['text_combined']
    analysis_label = df.iloc[analysis_email_index]['label']

    st.info(f"**Email Type:** {'🚨 Phishing' if analysis_label == 1 else '✅ Legitimate'}")

    # Display excerpt
    st.subheader("📄 Email Excerpt")
    st.text_area("Text:", analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text, height=100)

    # Tokenize and process
    cleaned_text = clean_text(analysis_text)
    tokens = tokenize_text(cleaned_text)
    tokens_no_stop = remove_stopwords(tokens)[:100]  # Limit for performance

    # Part-of-Speech Tagging
    st.subheader("🏷️ Part-of-Speech (POS) Tagging")

    st.markdown("""
    POS tagging identifies the grammatical role of each word (noun, verb, adjective, etc.)
    """)

    pos_tags = pos_tag(tokens_no_stop[:30])

    # Display POS tags
    pos_df = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
    pos_df.index = pos_df.index + 1

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(pos_df, use_container_width=True)

    with col2:
        st.markdown("""
        **Common POS Tags:**
        - NN: Noun
        - VB: Verb
        - JJ: Adjective
        - RB: Adverb
        - IN: Preposition
        - DT: Determiner
        """)

    # POS distribution
    pos_counts = Counter([tag for word, tag in pos_tags])

    fig, ax = plt.subplots(figsize=(10, 5))
    pos_items = list(pos_counts.items())
    pos_items.sort(key=lambda x: x[1], reverse=True)

    if pos_items:
        tags, counts = zip(*pos_items[:10])
        ax.bar(tags, counts, color='skyblue')
        ax.set_xlabel('POS Tag')
        ax.set_ylabel('Frequency')
        ax.set_title('Part-of-Speech Distribution (Top 10)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    # Named Entity Recognition
    st.subheader("🎯 Named Entity Recognition (NER)")

    st.markdown("""
    NER identifies and classifies named entities such as person names, organizations, locations, etc.
    """)

    # Process with NER
    try:
        ne_tree = ne_chunk(pos_tags)

        # Extract named entities
        entities = []
        for subtree in ne_tree:
            if hasattr(subtree, 'label'):
                entity_name = ' '.join([word for word, tag in subtree.leaves()])
                entity_type = subtree.label()
                entities.append((entity_name, entity_type))

        if entities:
            entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
            entity_df.index = entity_df.index + 1
            st.dataframe(entity_df, use_container_width=True)

            # Entity type distribution
            entity_types = Counter([entity[1] for entity in entities])
            if entity_types:
                fig, ax = plt.subplots(figsize=(8, 4))
                types, counts = zip(*entity_types.items())
                ax.bar(types, counts, color='coral')
                ax.set_xlabel('Entity Type')
                ax.set_ylabel('Count')
                ax.set_title('Named Entity Distribution')
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No named entities found in this sample.")
    except Exception as e:
        st.warning(f"NER analysis unavailable: {str(e)}")

    # N-gram Analysis
    st.subheader("📊 N-gram Analysis")

    st.markdown("""
    N-grams are contiguous sequences of n words. They help identify common phrases and patterns.
    """)

    ngram_type = st.radio("Select N-gram type:", ["Bigrams (2 words)", "Trigrams (3 words)"])

    if ngram_type == "Bigrams (2 words)":
        from nltk import bigrams
        ngrams_list = list(bigrams(tokens_no_stop))
        n = 2
    else:
        from nltk import trigrams
        ngrams_list = list(trigrams(tokens_no_stop))
        n = 3

    # Count n-grams
    ngram_freq = Counter([' '.join(ngram) for ngram in ngrams_list])
    top_ngrams = ngram_freq.most_common(15)

    if top_ngrams:
        # Display chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ngrams_labels = [ngram[0] for ngram in top_ngrams]
        ngrams_counts = [ngram[1] for ngram in top_ngrams]

        ax.barh(ngrams_labels, ngrams_counts, color='teal')
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top 15 {ngram_type}')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Display table
        ngram_df = pd.DataFrame(top_ngrams, columns=['N-gram', 'Frequency'])
        ngram_df.index = ngram_df.index + 1
        st.dataframe(ngram_df, use_container_width=True)
    else:
        st.info("Not enough tokens to generate n-grams.")

    # Vocabulary richness
    st.subheader("📚 Vocabulary Richness Analysis")

    col1, col2, col3 = st.columns(3)

    unique_words = len(set(tokens_no_stop))
    total_words = len(tokens_no_stop)

    if total_words > 0:
        lexical_diversity = unique_words / total_words

        with col1:
            st.metric("Total Words", total_words)
        with col2:
            st.metric("Unique Words", unique_words)
        with col3:
            st.metric("Lexical Diversity", f"{lexical_diversity:.3f}")

        st.markdown("""
        **Lexical Diversity** (Type-Token Ratio): Ratio of unique words to total words.
        - Higher values indicate more diverse vocabulary
        - Typical range: 0.4 - 0.8 for normal text
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📧 Phishing Email NLP Analysis Dashboard | Built with Streamlit</p>
    <p>Dataset: Phishing Email Dataset | NLP Libraries: NLTK, WordCloud</p>
</div>
""", unsafe_allow_html=True)

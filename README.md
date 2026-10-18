# CSEC 6: Natural Language Processing Projects

A collection of Natural Language Processing (NLP) projects completed for **CSEC 6 (Natural Language Processing)**. This repository covers end-to-end NLP workflows including modular token-level text normalization, statistical N-Gram language modeling, classical multi-class text classification, sparse vs. dense word embedding representations, and deep sequence modeling using **NLTK**, **Scikit-Learn**, **XGBoost**, **FastText**, **Hugging Face Datasets**, and **TensorFlow/Keras**.

> **Note on Datasets & Artifacts:** Raw dataset files (`.csv`), pretrained embedding archives (`cc.en.300.vec.gz`), serialized matrices (`.npz`, `.npy`, `.pkl`), and academic PDF reports are not committed to this repository for storage and privacy reasons. Direct links to the original datasets on **Kaggle** and **Hugging Face** are documented inside each Jupyter Notebook and in the sections below.

---

## Repository Structure

```text
csec6-projects/
├── README.md                                     # Project documentation, methodology & results
├── PS1_SpotifyReviews_TextNormalization.ipynb    # Problem Set 1: Modular Token-Level Preprocessing
├── PS2_HouseMD_NGramLanguageModeling.ipynb       # Problem Set 2: Unigram, Bigram & Trigram Models
├── PS3_AGNews_TextClassification.ipynb           # Problem Set 3: Multi-Class ML Text Classification
├── PS4_AGNews_TFIDF_vs_FastText.ipynb            # Problem Set 4: Sparse TF-IDF vs. Dense Embeddings
└── PS5_IMDB_RNN_vs_LSTM.ipynb                    # Problem Set 5: Sequence Modeling with RNN & LSTM
```

---

## 1. Problem Set 1: Data Acquisition and Text Normalization

### Overview & Dataset
This project implements a modular NLP preprocessing pipeline to normalize noisy user-generated app reviews using the **Spotify User Reviews Dataset**.
- **Dataset Scope:** Contains **52,686 non-null user reviews** (`52,702` total rows) scraped from the Google Play Store between January and July 2022, ranging from 3 to nearly 100 words per review. Sentiment labels consist of **44% Positive** (`23,263` instances) and **56% Negative** (`29,423` instances).
- **Dataset Access:** Available on Kaggle at [Spotify User Reviews Dataset](https://www.kaggle.com/datasets/alexandrakim2201/spotify-dataset) (`DATASET.csv`).

### Methodology
In contrast to the classical NLP pipeline (cleaning raw strings prior to tokenization), this implementation adopts a **modular token-level approach** where raw reviews are tokenized first and each token is transformed sequentially:
1. **Unitization & Tokenization:** Handled missing entries (`fillna('')`), dropped the sentiment `label` column, and segmented reviews into word and punctuation tokens using NLTK's `word_tokenize`.
2. **Case Normalization:** Converted all extracted tokens to lowercase.
3. **Numeric Token Removal:** Filtered out tokens containing numeric digits (`char.isdigit()`).
4. **Special Character & Whitespace Stripping:** Stripped non-alphanumeric characters from individual tokens via regular expressions (`re.sub(r'[^a-zA-Z0-9]', '', t)`) and discarded empty strings (e.g., converting `pop-up` to `popup` and stripping emojis/punctuation).
5. **Stopword Removal:** Removed standard English stopwords using `nltk.corpus.stopwords`.
6. **Stemming vs. Lemmatization:** Generated two parallel normalized representations from the cleaned tokens:
   - **`stemmed_review` (`PorterStemmer`):** Aggressively stripped suffixes (`e`, `y`, `ing`) for vocabulary compression and computational speed (e.g., `service` $\rightarrow$ `servic`, `quality` $\rightarrow$ `qualiti`, `easy` $\rightarrow$ `easi`, `spotify` $\rightarrow$ `spotifi`).
   - **`lemmatized_review` (`WordNetLemmatizer`):** Reduced inflectional forms to valid dictionary lemmas to preserve semantic accuracy (e.g., `stars` $\rightarrow$ `star`, `songs` $\rightarrow$ `song`, `books` $\rightarrow$ `book`).

---

## 2. Problem Set 2: N-Gram Language Model Construction

### Overview & Dataset
This project constructs statistical **Unigram, Bigram, and Trigram Language Models** from conversational medical drama dialogue using the **House M.D. Transcripts (Season 1)** dataset.
- **Dataset Scope:** Comprises **9,796 spoken dialogue lines** totaling **108,234 words** (excluding bracketed stage directions).
- **Dataset Access:** Available on Kaggle at [House M.D. Transcripts](https://www.kaggle.com/datasets/kunalbhar/house-md-transcripts/data) (`season1.csv`).

### Methodology
1. **Dialogue Cleaning & Normalization:** Stripped bracketed stage directions (`re.sub(r'\[.*?\]', '', str(x))`), dropped the speaker `name` column, and converted all text to lowercase.
2. **Sentence Segmentation & Boundary Marking:** Tokenized dialogue lines into individual sentences via `sent_tokenize`, stripped punctuation (`re.sub(r"[^a-z0-9\s]", "", sentence)`), and wrapped each sentence with start and end boundary tokens (`<s> ... </s>`) to distinguish intra-sentence transitions from cross-sentence boundaries.
3. **Corpus Flattening & N-Gram Counting:** Flattened all bounded sentence token lists into a sequential training corpus of **140,795 total tokens** (`16,509` sentences) and computed Unigram ($n=1$), Bigram ($n=2$), and Trigram ($n=3$) counts and Maximum Likelihood Estimate (MLE) conditional probabilities.

### Results & Findings

| Metric / Corpus Statistic | Value |
| :--- | :--- |
| **Raw Vocabulary Size (Whitespace Split)** | 15,683 unique words |
| **Final Vocabulary Size (Normalized Unigrams)** | 8,461 unique words |
| **Vocabulary Size Reduction** | **46.05%** |
| **Total Corpus Tokens (with `<s>` and `</s>`)** | 140,795 tokens |
| **Unique N-Grams (Unigrams / Bigrams / Trigrams)** | 8,461 / 49,911 / 88,375 |
| **Hapax Legomena (`count = 1`)** | 4,086 (48.3%) / 36,712 (73.6%) / 76,070 (86.1%) |

- **Frequent N-Gram Patterns:** Top lexical unigrams were dominated by conversational function words: `'the'` (`3,727`), `'you'` (`3,426`), `'to'` (`2,571`), `'a'` (`2,558`), and `'i'` (`2,346`). The most frequent sentence-initial bigram was `('<s>', 'i')` (`1,183` sentences starting with *"I"*), with notable bigram conditional probabilities including $P(\text{to} \mid \text{going}) = 0.7692$ (`76.92%`) and $P(\text{</s>} \mid \text{answer}) = 0.50$ (`50.0%`). In the trigram model, $P(\text{to} \mid \text{not going}) = 0.9091$ (`90.91%`) and $P(\text{you} \mid \text{why are}) = 0.8667$ (`86.67%`).
- **Combinatorial Explosion & Sparsity:** As $n$ increased from 1 to 3, unique n-grams expanded more than tenfold (`8,461` $\rightarrow$ `88,375`) while **86.08% of all trigrams** and **73.55% of all bigrams** occurred only once (hapax legomena), illustrating data sparsity in domain-specific dialogue and the necessity of smoothing or backoff techniques for unseen sequences.

---

## 3. Problem Set 3: Multi-Class Text Classification Analysis

### Overview & Dataset
This project benchmarks four supervised machine learning algorithms on high-dimensional, sparse text features using the **AG News Topic Classification Dataset**.
- **Dataset Scope:** Contains **120,000 training articles** and **7,600 testing articles** evenly distributed across 4 categories (`30,000` train / `1,900` test per class): **World (`0`)**, **Sports (`1`)**, **Business (`2`)**, and **Sci/Tech (`3`)**.
- **Dataset Access:** Loaded programmatically via Hugging Face `datasets` (`load_dataset("fancyzhx/ag_news")`). [Hugging Face Dataset Link](https://huggingface.co/datasets/fancyzhx/ag_news).

### Methodology
1. **Text Preprocessing:** Applied lowercase normalization, regex punctuation/special character removal (`re.sub(r"[^a-z0-9\s]", "", text)`), word-level tokenization (`word_tokenize`), and English stopword removal.
2. **TF-IDF Vectorization:** Constructed a sparse document-term matrix capped at the top **5,000 features** (`TfidfVectorizer(max_features=5000)`), producing feature matrices of shape `(120000, 5000)` for training and `(7600, 5000)` for testing.
3. **Model Evaluation:** Trained and compared **Multinomial Naive Bayes**, **Linear Support Vector Machine (`LinearSVC`)**, **Logistic Regression (`max_iter=1000`)**, and **XGBoost (`XGBClassifier`)** using macro-averaged accuracy, precision, recall, F1-score, and confusion matrices.

### Results & Model Comparison

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Notable Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **90.34%** (`0.9034`) | **0.9031** | **0.9034** | **0.9032** | Best overall performer; strong linear separability and interpretability |
| **Linear SVM (`LinearSVC`)** | **90.28%** (`0.9028`) | **0.9026** | **0.9028** | **0.9026** | Virtually tied with Logistic Regression on sparse TF-IDF features |
| **Multinomial Naive Bayes** | 88.89% (`0.8889`) | 0.8884 | 0.8889 | 0.8885 | Ultra-fast baseline; highest confusion between Business & Sci/Tech (`205` / `156`) |
| **XGBoost (Gradient Boosting)** | 88.58% (`0.8858`) | 0.8857 | 0.8858 | 0.8856 | High computational cost with no advantage over linear classifiers on sparse text |

---

## 4. Problem Set 4: Traditional Text Representations vs. Word Embeddings

### Overview & Dataset
Building on Problem Set 3, this study conducts a controlled comparison between **sparse lexical representations (TF-IDF)** and **dense pretrained word embeddings (FastText)** on the **AG News Dataset** (`120,000` train / `7,600` test samples across 4 news categories).

### Methodology
1. **Preprocessing & Vocabulary Extraction:** Cleaned and tokenized all articles (`105,133` unique tokens across the dataset).
2. **Feature Representation Pipelines:**
   - **TF-IDF Baseline (Sparse):** 5,000-dimensional sparse matrix (`99.6%` sparsity, `~34.0 MB` memory footprint) preserving explicit term-importance weights.
   - **FastText Plain Average (Dense):** Extracted `57,237` matching 300-dimensional vectors from pretrained FastText (`cc.en.300.vec.gz`) and computed the mean document vector (`300` dimensions, `0%` sparsity, `~288.0 MB` memory footprint).
   - **FastText TF-IDF Weighted Average (Dense + Weighting):** Weighted each token's 300-dimensional FastText embedding by its TF-IDF score prior to document-level averaging.
3. **Controlled Classification:** Evaluated all three representations using an identical `LogisticRegression(max_iter=1000)` classifier.

### Results & Representation Comparison

| Text Representation | Dimensions / Sparsity | Memory | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF (Sparse Baseline)** | `5,000` (`99.6%` sparse) | `~34.0 MB` | **90.26%** (`0.9026`) | **0.9023** | **0.9026** | **0.9024** |
| **FastText Plain Average** | `300` (`0.0%` sparse) | `~288.0 MB` | 88.32% (`0.8832`) | 0.8829 | 0.8832 | 0.8830 |
| **FastText TF-IDF Weighted** | `300` (`0.0%` sparse) | `~288.0 MB` | 87.49% (`0.8749`) | 0.8744 | 0.8749 | 0.8746 |

### Per-Class F1-Score Breakdown & Key Findings
- **Per-Class Performance (TF-IDF vs. FastText Plain vs. Weighted):**
  - **World:** `0.9079` vs. `0.8948` vs. `0.8857` (Top TF-IDF terms: *iraq, said, ap, us, reuters, president, minister*)
  - **Sports:** `0.9609` vs. `0.9495` vs. `0.9476` (Top TF-IDF terms: *game, win, team, season, first, cup, league, coach*)
  - **Business:** `0.8685` vs. `0.8360` vs. `0.8289` (Top TF-IDF terms: *us, oil, reuters, prices, company, percent, profit, stocks*)
  - **Sci/Tech:** `0.8723` vs. `0.8516` vs. `0.8361` (Top TF-IDF terms: *microsoft, new, software, internet, space, computer, technology*)
- **Why Sparsity Wins on Short News Texts:** TF-IDF outperformed FastText plain averaging by **1.94 percentage points** (`90.26%` vs. `88.32%`) and TF-IDF weighted averaging by **2.77 percentage points** (`87.49%`). Short news headlines and summaries depend heavily on sharp, category-specific lexical indicators; averaging static word embeddings across 300 dense dimensions smooths out and dilutes these discriminative keyword signals.

---

## 5. Problem Set 5: Sequence Modeling with RNN vs. LSTM

### Overview & Dataset
This project investigates sequential dependency modeling and long-range context retention in deep learning by comparing a vanilla **Recurrent Neural Network (`SimpleRNN`)** against a **Long Short-Term Memory (`LSTM`)** network on the **IMDB 50K Movie Reviews Dataset**.
- **Dataset Scope:** Contains **50,000 polar movie reviews** balanced equally between `positive` (`1`, 25,000) and `negative` (`0`, 25,000) sentiments. Partitioned via a stratified 80/20 split into **40,000 training** and **10,000 testing** samples (`SEED = 42`), with 20% of the training set (`8,000` samples) reserved for validation.
- **Dataset Access:** Available on Kaggle at [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`IMDB Dataset.csv`).

### Methodology
1. **Leakage-Free Sequence Preprocessing:** Stripped HTML tags (`<br />`) and non-alphanumeric characters, lowercased and tokenized text, fitted a Keras `Tokenizer(num_words=20000, oov_token="<OOV>")` strictly on the training partition, integer-encoded all reviews, and post-padded/truncated sequences to a uniform length of **200 tokens**.
2. **Controlled Neural Architectures:**
   - **Vanilla RNN (`2,572,417` params):** `Embedding(20000, 128, input_length=200)` $\rightarrow$ `SimpleRNN(64)` $\rightarrow$ `Dense(1, activation="sigmoid")`.
   - **LSTM (`2,609,473` params):** `Embedding(20000, 128, input_length=200)` $\rightarrow$ `LSTM(64)` $\rightarrow$ `Dense(1, activation="sigmoid")`.
3. **Training Configuration:** Both models were trained with the `adam` optimizer, `binary_crossentropy` loss, `batch_size=64`, a maximum of `5` epochs, and `EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)`.

### Results & Architectural Comparison

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test Loss | Best Val Accuracy | Confusion Matrix (`TN / FP / FN / TP`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleRNN (64 units)** | 50.13% (`0.5013`) | 0.5016 | 0.4202 | 0.4573 | 0.6952 | 52.10% (Epoch 3) | `TN: 2912` \| `FP: 2088`<br>`FN: 2899` \| `TP: 2101` |
| **LSTM (64 units)** | **77.46%** (`0.7746`) | **0.7741** | **0.7756** | **0.7748** | **0.5317** | **78.24%** (Epoch 5) | `TN: 3868` \| `FP: 1132`<br>`FN: 1122` \| `TP: 3878` |

### Sequence-Length Sensitivity Analysis

| Review Length Bucket (Tokens) | Test Samples | SimpleRNN Accuracy | LSTM Accuracy | LSTM Gain (`LSTM - RNN`) |
| :--- | :--- | :--- | :--- | :--- |
| **`<= 50` tokens** | 230 | 50.87% (`0.5087`) | **81.74%** (`0.8174`) | **+30.87%** |
| **`51 – 100` tokens** | 1,004 | 47.81% (`0.4781`) | **79.88%** (`0.7988`) | **+32.07%** |
| **`101 – 150` tokens** | 2,596 | 51.16% (`0.5116`) | **81.59%** (`0.8159`) | **+30.43%** |
| **`151 – 200` tokens** | 1,968 | 50.20% (`0.5020`) | **79.37%** (`0.7937`) | **+29.17%** |
| **`> 200` tokens** | 4,202 | 49.98% (`0.4998`) | **73.20%** (`0.7320`) | **+23.23%** |

- **Context Retention & Vanishing Gradients:** SimpleRNN failed to learn across 200-step sequences, stalling at chance-level accuracy (`50.13%`) as earlier token representations were repeatedly overwritten and gradients vanished across time steps. In contrast, LSTM's input, forget, and output gating mechanisms preserved long-range sentiment signals, boosting test accuracy by **+27.33 percentage points** (`77.46%`) and F1-score by **+31.75 percentage points** (`0.7748`).

---

## How to Run the Notebooks

1. Install the required Python dependencies:
   ```bash
   pip install numpy pandas scipy matplotlib nltk datasets scikit-learn xgboost tensorflow
   ```
2. Download the required datasets for local notebooks:
   - **Problem Sets 1, 2, and 5:** Download the CSV files from the Kaggle links in **Sections 1, 2, and 5** (`DATASET.csv`, `season1.csv`, and `IMDB Dataset.csv`) and place them in the repository root.
   - **Problem Sets 3 and 4:** The AG News dataset (`fancyzhx/ag_news`) and FastText vectors (`cc.en.300.vec.gz`) are downloaded automatically inside the notebooks via `datasets` and `urllib`.
3. Launch Jupyter Notebook or open the `.ipynb` files in VS Code / Google Colab:
   ```bash
   jupyter notebook
   ```

---

## Author
- **Jessica Mae T. Lanuzo** — BSCS

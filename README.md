# zenodo-PAN22-aut-prof-irony-stereotype
 PAN 22 Author Profiling: Profiling Irony and Stereotype Spreaders on Twitter (IROSTEREO)

### Link to Task:
https://pan.webis.de/clef22/pan22-web/author-profiling.html

## Features Used:
- Vader Sentiment Standard Deviation (Across Tweets)
- Vocab Size
- Type/Token Ration (N_Vocabs/N_Words)
- Avg. Word Length
- Avg. Tweet Length
- Repeated Punctuations Count "..." "?!" "!" "?"
- Emoji Counts (Using TF-IDF + PCA) 
- Profanity Counts (Using TF-IDF + PCA) 
- Word Counts (Using TF-IDF + PCA) 
- Avg. Proportion of Capital Letters
- Avg. Tweet LiX complexity
- Counts of POS tags (Avg. Tweet)

## Results:
**Average F1 Score on Cross-Validation (70% Train), RandomForest:** 91.83%

**F1 Score on 30% Train (Test Split), RandomForest:** 96.04%

## Usage Example:
You can check the file _test_predictions.py_ for a simple case of loading features and making predictions.

    # Import necessary files
    from utils import *
    from classifier_methods import *

    # Load data
    X, y, USERCODE_X, lang = load_dataset(os.path.join(os.getcwd(),"data","en"))
    X_test, USERCODE_TEST_X, lang_test = load_dataset(os.path.join(os.getcwd(),"data","test","en"), is_test=True)
    # Train a model (default is Random Forest)
    classifier, *settings = train_model(X,y)
    # Generate test features
    X_test_features = get_features_test(X_test, *settings)
    # Get Predictions:
    pred = classifier.predict(X_test_features)


## Features Not Implemented:
- Syntactic Complexity
- Number of Typos - Implemented, but not very useful.

## Possible Literature:
- https://dl.acm.org/doi/epdf/10.1145/2930663 (Irony Detection in Twitter)

### Interesting Tools:
- https://github.com/twintproject/twint

### Google Docs:
- https://docs.google.com/document/d/1gfvLkT3j-7GNFuwSzb1rLRq5NNAaPcGRwPfgnSKnpOc/edit?usp=sharing

#### Possible Features:
- Polarity Score - Emotions/Semantics
- Counts of Emojis 
- Set-up Sentences
- Language Complexity (LiX)
- Typos (Minimum Edit Distance)
- n-grams
- Stylistic features
- Semantic polarity 
- POS tags
- Lemma
- Profanity
- Punctuation Features - Look for ("Funny") - Proportions of punctuation 
- Proportion of capital letters per word - Capture ThIs SpEaCh
- Label each sentence and set a threshold for Irony spreads
- Unexpectedness
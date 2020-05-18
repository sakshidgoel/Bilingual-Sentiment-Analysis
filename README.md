# Bilingual-Sentiment-Analysis
## Sentiment Analysis of Hindi-English Code-mixed Social Media Text
The main aim of the project is to develop a sentiment analyzer that can be used on twitter data to classify it as positive or negative. Our project takes care of the challenge of bilingual comments, where people tweet in two languages, in this case Hindi and English, in the Latin Alphabet. 

## Introduction
This project discusses the different classifiers that can be used for sentiment analysis of twitter data, to classify the tweets as positive or negative. The challenge of Hindi-English Code-mixed Social Media Text is focused on here. An existing labelled Kaggle dataset is used for this study. Forty thousand rows of this dataset are randomly selected and then cleaned. Adjectives, adverbs and abstract nouns are selected as features and extracted for each cleaned tweet. Then seven different classifiers, namely, **Naïve Bayes’**, **Multinomial Naïve Bayes’**, **Bernouille’s Naïve Bayes’**, **Logistic Regression**, **Stochastic Gradient Descent**, **Support Vector Machines** and **Maximum Entropy classifiers** are trained on 85% of the dataset. A **hybrid model** is created using these seven classifiers by implementing the voting based ensemble model. Then a function is created which uses **TextBlob** to identify the language of the word and in case, the language is ‘hi’, i.e. Hindi, then the **Google Machine Translator** is used to convert that word to its English form. The function also tries to handle the challenge of language ambiguity which occurs when a word exists in both the English and Hindi dictionaries. The translated final string is passed to another function that uses the string as a test case for the seven base classifiers and the hybrid model and the sentiment predicted  by each of these classifiers is printed along with the extracted features and the translated final tweet/string given by the user. 

## What is Bilingual Sentiment Analysis?
Bilingual Sentiment Analysis is a subset of multilingual sentiment analysis which is one of many popular types of sentiment analysis. It handles the sentiment analysis of text that includes more than one language (in the case of bilingual, two languages, i.e. English and Hindi here)

## What is Code Mixing?
Code in sociolinguistics refers to a language or a language variety. Code-mixing can be defined as simply mixing of two or more varieties of the same language or of different languages altogether. Example of Code-Mixed Hindi-English text is – “tu apne saath college bag leja raha hai?” or “arey waah! I am very proud of you”. 

## Dataset
The dataset that was used was obtained from “Kaggle” called the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140). It contains 1,600,000 tweets extracted using the twitter API. The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment. But only 40000 rows where randomly selected from this dataset, with equal distribution of positive and negative tweets, i.e. neutral tweets were ignored as this study focuses on binary classification. 

## Data Preprocessing
In order to make sure that the work was carried out in the most efficient manner, the dataset used needed to be preprocessed before actual usage.
### Data Cleaning
1) First, only a handful of columns were used from the set of 6 columns. These columns used were *Tweet* and *Label*. The remaining columns were discarded as they were not needed.  
2) Next, any HTML tags present in the text were removed with the help of BeautifulSoup package. Quotation marks ( ‘, “) and URLs were removed as they are unnecessary for the sentiment analysis of the text. If any emoticons were used in the tweet, they were converted into their equivalent emotion that they are intended to signify, and the emojis were removed.
3) Then, all the words in the text are converted to lower case and repeating words in the same tweet are removed. 
4) Concatenated words, i.e. words which were joined such as “Can’t” were expanded to “Can not”.
5) The data cleaning process also included the removal of punctuations, numbers, special characters and user handles or tags (“@” symbol followed by the account handle). Stopwords, other than ‘not’ and the word ‘is’ were also removed from the text. 
6) The cleaned tweet contained corrected spellings of all words, because of the use of the ‘SpellChecker’ package. All words with length equal to one were also removed. 
7) Next, the words were lemmatized, and words preceded  by ‘not’ were substituted by their antonyms to represent the negation effect. 
Finally, the dataset contained the cleaned tweets.
### Feature Extraction
Certain features, like adjectives, abstract nouns, and adverbs were focused on and the rest of the words were removed as they did not add any value to the sentiment. This was done as part for the feature identification and extraction. 
This was implemented by checking each word in the cleaned tweet with the words in a file, which was prefilled with most common adjectives, adverbs and abstract nouns. If the word was not present in this file, it was not chosen as a feature and if it matched a word in this file, it was selected as one of the features. 
All these features of each text was stored in another column. This method was used because the existing method is not completely accurate and is outdated. For example, the word ‘clever’ was being identified as a ‘noun’ by the pos_tag function provided by the nltk library.
### Final Dataset
This dataset has three columns, namely *Tweet*, *Label* and *cleaned_tweets*. The first two are the ones taken from the original dataset. The last column is a derived column from all the preprocessing and contains just the features extracted from each of the tweet texts, as string. 

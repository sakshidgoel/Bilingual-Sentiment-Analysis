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

## Proposed system
Once we have the pre-processed dataset, this dataset is split into training and testing data with a ratio of 85%. This approximates to 34000 tweets in training data and 6000 tweets in testing data. Then the above mentioned seven classifiers are trained on this data and then tested against the testing data and the performance of these classifiers is evaluated. Next a hybrid model as a voting based ensemble model of these seven classifiers is constructed. Following suit, this model is trained and then tested to evaluate its performance. After this, a translation mechanism is used to handle the challenge of the Hinglish text using the Google Translator Machine and a function is created that takes in text as input and translates it if required and then uses the seven trained base classifiers and the hybrid model to predict the sentiment of the input.

## What is a hybrid model?
Hybrid machine learning models combine strengths of various models or classifiers and bring the best from them. Through this hybrid machine, it is intended to improve the classification accuracy of the classifiers. 

## What is an ensemble model? 
+ Ensemble models in machine learning combine the decisions from multiple models to improve the overall performance. They are a divide and conquer approach. 
+ Ensemble modelling is a process where multiple diverse base models are used to predict an outcome. The approach seeks the wisdom of the crowds in making a prediction. 
+ Even though the ensemble model has multiple based models (seven models here), it acts and performs as a single model.
+ Voting based ensemble model: This is a type of ensemble model where the mode of the predicted labels is used as the final classification.  

## Why Ensemble Models?
+ The motivation for using ensemble models is to reduce the generalization error of the prediction. 
+ As long as the base models are diverse and independent, the prediction error decreases when the ensemble approach is used. 
+ The main causes of error in learning models are due to noise, bias and variance. Ensemble models help to minimize these factors. 
+ These methods are designed to improve the stability and the accuracy of machine learning algorithms and classifiers. 

## Performance
To evaluate the performance of all of the classifier models, the precision, recall and F1-Score. The metrics are as follows:

### Naive Bayes' 
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Naive-Bayes'.png" width="500">

### Bernouille's Naive Bayes'
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Bernouille's-Naive-Bayes'.png" width="500">

### Multinomial Naive Bayes'
    
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Multinomial-Naive-Bayes'.png" width="500">

### Logistic Regression
    
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Logistic-Regression.png" width="500">

### Stochastic Gradient Descent
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Stochastic-Gradient-Descent.png" width="500">

### Support Vector Machine
    
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Support-Vector-Machine.png" width="500">

### Maximum Entropy
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Maximum-Entropy.png" width="500">

### Hybrid Model
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Hybrid-Model.png" width="500">

## Accuracy
     
The graph showing comparison of the accuracies of each of the classifiers is as follows:  
    
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Accuracies-Graph.png" width="500">
   
As it can be observed, the hybrid model performs the best out of all the classifiers, fulfilling its purpose.

## Confusion Matrix
The confusion matrix for the hybrid model before and after normalization have been shown below:

### Confusion Matrix Data
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Confusion%20matrix%20data.png" width="500">

### Confusion Matrix before normalization
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/confusion_matrix_before_normalization.png" width="500">

### Confusion Matrix after normalization
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/confusion_matrix_after_normalization.png" width="500">

## Snapshots of few test cases

### Test Case 1
Text: *arrey waah! I'm very proud of you*  
      
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Test1.png" width="500">

### Test Case 2
Text: *tum log pagal ho*   
            
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Test2.png" width="440">

### Test Case 3
Text: *tum pagal ho*   
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Test3.png" width="430">

### Test Case 4
Text: *tum kharab ho*   
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Test4.png" width="420">

### Test Case 5
Text: *you are not accha*   
     
<img src = "https://github.com/sakshidgoel/Bilingual-Sentiment-Analysis/blob/master/Screenshots/Test5.png" width="440">

## Reference Papers
1) Aditya Joshi, Ameya Prabhu Pandurang, Manish Shrivatsava and Vasudeva Varma, “Towards Sub-Word Level Compositons for Sentiment Analysis of Hindi-English Code Mixed Text,” 26th International Conference on Computational Linguistics, December 2017. 
2) R. Mahesh, K. Sinha and Anil Thakur, “Machine Translation of Bilingual Hindi-English (Hinglish) Text,” January 2005.
3) Aditya Bohra, Deepanshu Vijay, Vinay Singh, Syed S. Akhtar and Manish Shrivatsava, “A Dataset of Hindi-English Code-Mixed Social Media Text for Hate Speech Detection,” 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics, June 2018. 
4) Sakshi Gupta, Piyush Bansal and Radhika Mamidi, “Resource Creation for Hindi-English Code Mixed Social Media Text,” July 2016. 
5) Kumar Ravi and Vadlamani Ravi, “Sentiment classification of  Hinglish text,” March 2016.

## Reference Links
1) [Sentiment Analysis](https://monkeylearn.com/sentiment-analysis/)
2) [Code Mixing](http://languagelinguistics.com/2017/06/27/code-mixingsociolinguistics/)
3) [Machine Learning Concepts](https://deepai.org/machine-learning-glossary-and-terms/classifier)
4) [Machine Learning Classifiers](https://towardsdatascience.com/machine-learning-classifiersa5cc4e1b0623)
5) [Classification Algorithms](https://analyticsindiamag.com/7-types-classification-algorithms/)
6) [Logistic Regression](https://dataaspirant.com/2017/03/02/how-logistic-regression-modelworks/)
7) [Naive Bayes'](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes)
8) [Ensemble Learning](https://www.datacamp.com/community/tutorials/ensemble-learningpython)
9) [Ensemble Classifier](https://towardsdatascience.com/advanced-ensemble-classifiers8d7372e74e40)
10) [Confusion Matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)

## Contact
For any comments or questions, please contact us at sakshidgoel@gmail.com / suhailrahman8999@gmail.com

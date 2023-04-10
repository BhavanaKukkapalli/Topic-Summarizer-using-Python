# Text-Summarizer-using-Python

## Project Summary
The purpose of this project is to provide a solution for undergraduate students to easily understand unfamiliar topics in a short amount of time. The solution involves web scraping data from trusted sources and summarizing the data using machine learning algorithms. The system architecture and methodology are discussed in detail along with implementation and future work.

## Problem Statement
Undergraduate students struggle to understand vast amounts of data in a short amount of time, and identifying correct information is a major problem. The project aims to provide a solution to this problem by summarizing the required topic from trusted sources and websites.

## Specific Objectives
The specific objectives of this project are:
1. Web scraping data.
2. Summarizing the data using machine learning algorithms, which includes cleaning the data, POS tagging, and similarity matrix.

## Findings
The following findings were discovered during the project:
1. Graph-based text summarization is the most suitable and efficient approach for this application.
2. POS tagging involves mapping tokens with corresponding parts of speech and resolving pronouns into corresponding nouns.
3. A comparative study of various similarity matrices was conducted to identify the best one for the system.

## System Specification
The system requirements for running the project are:
1. A system with i3 or i5 processor.
2. RAM with 4GB or higher.
3. Any running version of a text editor.
4. System with installed Python libraries.
5. A system with a good network connection.

## Methodology
The methodology for the project involves five different stages of graph-based text summarization:
1. Tokenization
  a. Tokenization of the paragraph is done in two different stages to finally get words(tokens).
  b. Sentence tokenizing is used to break paragraphs down into sentences.
  c. Word tokenizing is used to break sentences down into words or tokens.
2. POS Tagging
  a. Tokens are mapped with corresponding parts of speech.
3. Pronoun Resolution
  a. Pronouns are resolved into corresponding nouns, which can be objects or persons.
4. Graph Building
  a. A graph is constructed with nouns as vertices and edges are given weights that correspond to the relevance between the nouns.
5. Sentence Equivalence
  a. Sentences are scored based on the collective weights of nouns in the sentence.
  b. The sentence with the highest score is chosen for the final summary.

## Implementation
The implementation process for the project is as follows:

1. Enter the topic into the search box.
2. Data is web scraped and merged into a single document.
3. Data is tokenized using NLTK library, and data cleaning is performed to remove punctuation and special characters.
4. POS tagging is done, followed by Pronoun Resolution and stop word removal.
5. Word embedding and sentence vectors are formed to find the similarity between sentences, and text summarization is performed.
6. The summary is available for viewing and downloading.

## Survey Results
To confirm the accuracy of the algorithm, respective heat maps have been projected. The results from a survey conducted on a sample of 25 users were also taken into account to arrive at the most efficient way to choose which similarity matrix can be used for the system.

## Future Work
Future work for the project involves improving the accuracy of the algorithm, increasing the speed of web scraping, and expanding the sources for web scraping. The project can also be extended to include audio and video summarization.

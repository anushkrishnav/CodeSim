# Code Similarity Using Sequence Models

## Introduction
The idea of the project is to try out different existing sequence models to find out the similarity between two code snippets. 
for example 
| Code 1 | Code 2 | Similarity |
|--------|--------|------------|
| `def add(a, b): return a + b` | `def sum(x, y): return x + y` |? |

The base idea is to get the embeddings of the code snippets and then find the similarity between them. by calculating the cosine similarity between the embeddings of the code snippets.

## Before we start
Code Embeddings :
- Code embeddings are simply the way to transform the code as seen in the above example into a dense vector in a high-dimensional space. 
- Basically finding it a place in a high-dimensional space where similar code snippets are close to each other for example like your house in the city which is your place in the city where you live and you are close to your friends and family.
- The reason we need code embeddings is that we can't directly compare the code snippets as they are in the form of text and we need to convert them into a form where we can compare them. by creating embeddings of code we try to caputre the semantics, meaning, and context of the code. 

## How do we create Code embeddings ?
The idea of this project is to explore the multiple way to do multiple things to come up with a satisfactory solution that will enable me to compare 2 code snippets and find out the similarity between them.

The commons ways to create code embeddings are:
- Treat code as a text sequence and use Pre-trained language models like BERT, GPT-2, etc.
- Treat code as a graph and use Graph Neural Networks to create embeddings.

## Process

### Dataset 
Apart from my personal collection of methods across different python repos, I will be using the PoolC/5-fold-clone-detection-600k-5fold dataset which is a dataset of 600k code snippets with their corresponding labels. 

### Preprocessing
- Extracting the dpcstring and docstring from the code snippets.
- Copying method definitions to a new column as a new feature.
- Cleaning the code snippets by removing the comments, docstrings, and other unnecessary information. 
- Tokenizing the code snippets.
- Padding the code snippets to make them of the same length ( for some models)
- Creating the embeddings of the code snippets.
- Calculating the cosine similarity between the embeddings of the code snippets.
- Evaluating the model on the test set.
- Comparing the results of different models.

## Embedding Models 
- Word2Vec (Interpretable)
- Code2Vec 
- Transformer based (CodeBERT, GPT)
- OpenAI ada-002 Code Embedding Model
- OpenAI text-embedding-3-small

## Evaluation
- Cosine Similarity
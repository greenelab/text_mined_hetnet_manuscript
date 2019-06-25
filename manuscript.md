---
author-meta:
- David N. Nicholson
date-meta: '2019-06-25'
keywords:
- machine learning
- weak supervision
- natural language processing
- heterogenous netowrks
lang: en-US
title: Mining Heterogenous Relationships from Pubmed Abstracts Using Weak Supervision
...






<small><em>
This manuscript
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/daac04685c004a220521d1e948777cabfef6e7ea/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@daac046](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/daac04685c004a220521d1e948777cabfef6e7ea)
on June 25, 2019.
</em></small>

## Authors



+ **David N. Nicholson**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon}
    [0000-0003-0002-5761](https://orcid.org/0000-0003-0002-5761)
    · ![GitHub icon](images/github.svg){.inline_icon}
    [danich1](https://github.com/danich1)
    · ![Twitter icon](images/twitter.svg){.inline_icon}
    [N/A](https://twitter.com/N/A)<br>
  <small>
     Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania
     · Funded by GBMF 4552
  </small>



## Abstract {.page_break_before}

This is a **rough draft** of a manscript on label function reuse for text mining heterogenous relationship from Pubmed Abstracts.


#Introduction
Set introduction for paper here
Talk about problem, goal, and significance of paper

## Recent Work
Talk about what has been done in the field in regards to text mining and knowledge base integration


#Materials and Methods

##Dataset
Talk about dataset - Pubtator
Talk about preprocessing Pubtator
Talk about hand annotations for each realtion

## Label Functions
describe what a label function is and how many we created for each relation

## Training Models
### Generative Model
talk about generative model and how it works
### Word Embeddings
mention facebooks fasttext model and how we used it to train word vectors
### Discriminator Model
talk about the discriminator model and how it works
### Discriminator Model Calibration
talk about calibrating deep learning models with temperature smoothing

## Experimental Design
talk about sampling experiment


# Results

## Random Sampling of Generative Model
place the grid aurocs here for generative model

## Discriminator Model Builds Off Generative Model
place the grid of aurocs here for discriminator model

## Random Noise Generative Model
place the results of random label function experiment

## Reconstructing Hetionet
place figure of number of new edges that can be added to hetionet as well as edges we can reconstruct using this method


# Discussion
Here mention why performnace increases in the beginning for the generative model then decreases

Discuss discriminator model performance given generative model

Mention Take home messages

1. have a centralized set of negative label functions and focus more on contstructing positive label functions


# Conclusion and Future Direction
Recap the original problem - takes a long time to create useful label function

Proposed solution - reuse label functions

Mention incorporating more relationships
Mention creating a centralized multitask text extractor using this method.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

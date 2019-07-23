---
author-meta:
- David N. Nicholson
- Daniel S. Himmelstein
- Casey S. Greene
date-meta: '2019-07-23'
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
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/0a7be3e9a4633dfcb49e9d734c97926d40786ac9/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@0a7be3e](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/0a7be3e9a4633dfcb49e9d734c97926d40786ac9)
on July 23, 2019.
</em></small>

## Authors



+ **David N. Nicholson**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon}
    [0000-0003-0002-5761](https://orcid.org/0000-0003-0002-5761)
    · ![GitHub icon](images/github.svg){.inline_icon}
    [danich1](https://github.com/danich1)<br>
  <small>
     Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania
     · Funded by GBMF4552
  </small>

+ **Daniel S. Himmelstein**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon}
    [0000-0002-3012-7446](https://orcid.org/0000-0002-3012-7446)
    · ![GitHub icon](images/github.svg){.inline_icon}
    [dhimmel](https://github.com/dhimmel)
    · ![Twitter icon](images/twitter.svg){.inline_icon}
    [dhimmel](https://twitter.com/dhimmel)<br>
  <small>
     Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania
     · Funded by GBMF4552
  </small>

+ **Casey S. Greene**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon}
    [0000-0001-8713-9213](https://orcid.org/0000-0001-8713-9213)
    · ![GitHub icon](images/github.svg){.inline_icon}
    [cgreene](https://github.com/cgreene)
    · ![Twitter icon](images/twitter.svg){.inline_icon}
    [GreeneScientist](https://twitter.com/GreeneScientist)<br>
  <small>
     Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania
     · Funded by GBMF4552 and R01 HG010067
  </small>



## Abstract {.page_break_before}

This is a **rough draft** of a manscript on label function reuse for text mining heterogenous relationship from Pubmed Abstracts.


## Introduction

Knowledge bases are important resources that hold complex structured and unstructed information. 
These resources have been used in important tasks such as network analysis for drug repurposing discovery [@u8pIAt5j; @bPvC638e; @O21tn8vf] or as a source of training labels for text mining systems [@EHeTvZht; @CVHSURuI; @HS4ARwmZ]. 
Populating knowledge bases often requires highly-trained scientists to read biomedical literature and summarize the results [@N1Ai0gaI].
This manual curation process requires a significant amount of effort and time: in 2007 researchers estimated that filling in the missing annotations at that point would require approximately 8.4 years [@UdzvLgBM]).
The rate of publications has continued to increase exponentially [@1DBISRlwN].
This has been recognized as a considerable challenge and leads to gaps in knowledge bases [@UdzvLgBM].  
Relationship extraction has been studied as a solution towards handling this problem [@N1Ai0gaI].
This process consists of creating a machine learning system to automatically scan and extract relationships from textual sources.
Machine learning methods often leverage a large corpus of well-labeled training data, which still requires manual curation.
Distant supervision is one technique to sidestep the requirement of well-annotated sentences: with distant supervision one makes the assumption that that all sentences containing an entity pair found in a selected database provide evidence for a relationship [@EHeTvZht].
Distant supervision provides many labeled examples; however it is accompanied by a decrease in the quality of the labels.  
Ratner et al. [@5Il3kN32] recently introduced "data programming" as a solution.
Data programming combines distant supervision with the automated labeling of text using hand-written label functions.
The distant supervision sources and label functions are integrated using a noise aware generative model, which is used to produce training labels.
Combining distant supervision with label functions can dramatically reduce the time required to acquire sufficient training data.
However, constructing a knowledge base of heterogeneous relationships through this framework still requires tens of hand-written label functions for each relationship type.
Writing useful label functions requires significant error analysis, which can be a time-consuming process.  

In this paper, we aim to address the question: to what extent can label functions be re-used across different relationship types?
We hypothesized that sentences describing one relationship type may share information in the form of keywords or sentence structure with sentences that indicate other relationship types.
We designed a series of experiments to determine the extent to which label function re-use enhanced performance over distant supervision alone.
We examine relationships that indicate similar types of physical interactions (i.e., gene-binds-gene and compound-binds-gene) as well as different types (i.e., disease-associates-gene and compound-treats-disease).
The re-use of label functions could dramatically reduce the number required to generate and update a heterogeneous knowledge graph.

## Recent Work

Talk about what has been done in the field in regards to text mining and knowledge base integration


<style> 
span.gene_color { color:#02b3e4 } 
span.disease_color { color:#875442 }
</style>

# Materials and Methods

## Hetionet
Hetionet [@O21tn8vf] is a large heterogenous network that contains pharmacological and biological information.
This network depicts information in the form of nodes and edges of different types: nodes that represent biological and pharmacological entities and edges which represent relationships between entities. 
Hetionet v1.0 contains 47,031 nodes with 11 different data types and 2,250,197 edges that represent 24 different relationship types (Figure {@fig:hetionet}).
Edges in Hetionet were obtained from open databases, such as the GWAS Catalog [@16cIDAXhG] and DrugBank [@16cIDAXhG].
For this project, we analyzed performance over a subset of the Hetionet relationship types: disease associates with a gene (DaG), compound binds to a gene (CbG), gene interacts with gene (GiG) and compound treating a disease (CtD).

![
A metagraph (schema) of Hetionet where pharmacological, biological and disease entities are represented as nodes and the relationships between them are represented as edges.
This project only focuses on the information shown in bold; however, we can extend this work to incorporate the faded out information as well.
](images/figures/hetionet/metagraph_highlighted_edges.png){#fig:hetionet}


## Dataset
We used PubTator [@13vw5RIy4] as input to our analysis.
PubTator provides MEDLINE abstracts that have been annotated with well-established entity recognition tools including DNorm [@vtuZ3Wx7] for disease mentions, GeneTUKit [@4S2HMNpa] for gene mentions, Gnorm [@1AkC7QdyP] for gene normalizations and a dictionary based look system for compound mentions [@r501gnuM].
We downloaded PubTator on June 30, 2017, at which point it contained 10,775,748 abstracts. 
Then we filtered out mention tags that were not contained in hetionet.
We used the Stanford CoreNLP parser [@RQkLuc5t] to tag parts of speech and generate dependency trees.
We extracted sentences with two or more mentions, termed candidate sentences.
Each candidates sentence was stratified by co-mention pair to produce a training set, tuning set and a testing set (shown in Table {@tbl:candidate-sentences}).
Each unique co-mention pair is sorted into four categories: (1) in hetionet and has sentences, (2) in hetionet and doesn't have sentences, (3) not in hetionet and does have sentences and (4) not in hetionet and doesn't have sentences.
Within these four categories each pair receives their own individual partition rank (continuous number between 0 and 1).
Any rank lower than 0.7 is sorted into training set, while any rank greater than 0.7 and lower than 0.9 is assigned to tuning set.
The rest of the pairs with a rank greater than or equal to 0.9 is assigned to the test set.
Sentences that contain more than one co-mention pair are treated as multiple individual candidates.
We hand labeled five hundred to a thousand candidate sentences of each relationship to obtain to obtain a ground truth set (Table {@tbl:candidate-sentences}, [dataset](http://github.com/text_minded_hetnet_manuscript/master/supplementary_materials/annotated_sentences)).

| Relationship | Train | Tune | Test |
| :--- | :---: | :---: | :---: |
| Disease Associates Gene | 2.35 M |31K (397+, 603-) | 313K (351+, 649-) |
| Compound Binds Gene | 1.7M | 468K (37+, 463-) | 227k (31+, 469-) |
| Compound Treats Disease | 1.013M | 96K (96+, 404-) | 32K (112+, 388-) |
| Gene Interacts Gene | 12.6M | 1.056M (60+, 440-) | 257K (76+, 424-) |

Table: Statistics of Candidate Sentences. 
We sorted each candidate sentence into a training, tuning and testing set.
Numbers in parentheses show the number of positives and negatives that resulted from the hand-labeling process.
{#tbl:candidate-sentences}

## Label Functions for Annotating Sentences
A common challenge in natural language processing is having too few ground truth annotations, even when textual data are abundant.
Data programming circumvents this issue by quickly annotating large datasets by using multiple noisy signals emitted by label functions [@5Il3kN32].
Label functions are simple pythonic functions that emit: a positive label (1), a negative label (-1) or abstain from emitting a label (0).
We combine these functions using a generative model to output a single annotation, which is a consensus probability score bounded between 0 (low chance of mentioning a relationship) and 1 (high chance of mentioning a relationship).
We used these annotations to train a discriminator model that makes the final classification step.
Our label functions fall into three categories: databases, text patterns and domain heuristics.
We provide examples for the categories, described below, using the following candidate sentence: "[PTK6]{.gene_color} may be a novel therapeutic target for [pancreatic cancer]{.disease_color}."

**Databases**: These label functions incorporate existing databases to generate a signal, as seen in distant supervision [@EHeTvZht].
These functions detect if a candidate sentence's co-mention pair is present in a given database.
If the pair is present, emit a positive label and abstain otherwise.
If the pair isn't present in any existing database, then a separate label function will emit a negative label.
We use a separate label function to prevent the label imbalance problem. This problem occurs when candidates, that scarcely appear in databases, are drowned out by negative labels.
The multitude of negative labels increases the likelihood of misclassification when training the generative model.

$$ \Lambda_{DB}(\color{#875442}{D}, \color{#02b3e4}{G}) = 
\begin{cases}
 1 & (\color{#875442}{D}, \color{#02b3e4}{G}) \in DB \\
0 & otherwise \\
\end{cases} $$

$$ \Lambda_{\neg DB}(\color{#875442}{D}, \color{#02b3e4}{G}) = 
\begin{cases}
 -1 & (\color{#875442}{D}, \color{#02b3e4}{G}) \notin DB \\
0 & otherwise \\
\end{cases} $$

**Text Patterns**: These label functions are designed to use keywords and sentence context to generate a signal. 
For example, a label function could focus on the number of words between two mentions or focus on the grammatical structure of a sentence.
These functions emit a positive or negative label depending on the situation.
In general, those focused on keywords emit positives and those focused on negation emit negatives.

$$ \Lambda_{TP}(\color{#875442}{D}, \color{#02b3e4}{G}) = 
\begin{cases}
 1 & "target" \> \in Candidate \> Sentence \\
 0 & otherwise \\
\end{cases} $$

$$ \Lambda_{TP}(\color{#875442}{D}, \color{#02b3e4}{G}) = 
\begin{cases}
 -1 & 	"VB" \> \notin pos\_tags(Candidate \> Sentence) \\
 0 & otherwise \\
\end{cases} $$


**Domain Heuristics**: These label functions use the other experiment results to generate a signal. 
For this category, we used dependency path cluster themes generated by Percha et al [@CSiMoOrI].
If a candidate sentence's dependency path belongs to a previously generated cluster, then the label function will emit a positive label and abstain otherwise.

$$
\Lambda_{DH}(\color{#875442}{D}, \color{#02b3e4}{G}) = \begin{cases}
    1 & Candidate \> Sentence \in Cluster \> Theme\\
    0 & otherwise \\
    \end{cases}
$$

Roughly half of our label functions are based on text patterns, while the others are distributed across the databases and domain heuristics (Table {@tbl:label-functions}).

| Relationship | Databases (DB) | Text Patterns (TP) | Domain Heuristics (DH) |
| --- | :---: | :---: | :---: |
| Disease associates Gene (DaG) | 7 | 20 | 10 | 
| Compound treats Disease (CtD) | 3 | 15 | 7 |
| Compound binds Gene (CbG) | 9 | 13 | 7 | 
| Gene interacts Gene (GiG) | 9 | 20 | 8 | 

Table: The distribution of each label function per relationship. {#tbl:label-functions} 

## Training Models
### Generative Model
The generative model is a core part of this automatic annotation framework.
It integrates multiple signals emitted by label functions and assigns a training class to each candidate sentence.
This model assigns training classes by estimating the joint probability distribution of the latent true class ($Y$) and label function signals ($\Lambda$), $P(\Lambda, Y)$.
Assuming each label function is conditionally independent, the joint distribution is defined as follows:  

$$
P(\Lambda, Y) = \frac{\exp(\sum_{i=1}^{m} \theta^{T}F_{i}(\Lambda, y))}
{\sum_{\Lambda'}\sum_{y'} \exp(\sum_{i=1}^{m} \theta^{T}F_{i}(\Lambda', y'))}
$$  

where $m$ is the number of candidate sentences, $F$ is the vector of summary statistics and $\theta$ is a vector of weights for each summary statistic.
The summary statistics used by the generative model are as follows:  

$$F^{Lab}_{i,j}(\Lambda, Y) = \unicode{x1D7D9}\{\Lambda_{i,j} \neq 0\}$$
$$F^{Acc}_{i,j}(\Lambda, Y) = \unicode{x1D7D9}\{\Lambda_{i,j} = y_{i,j}\}$$   

*Lab* is the label function's propensity (the frequency of a label function emitting a signal).
*Acc* is the individual label function's accuracy given the training class.
This model optimizes the weights ($\theta$) by minimizing the negative log likelihood:

$$\hat{\theta} = argmin_{\theta} -\sum_{\Lambda} log \sum_{Y}P(\Lambda, Y)$$

In the framework we used predictions from the generative model, $\hat{Y} = P(Y \mid \Lambda)$, as training classes for our dataset [@9Jo1af7Z; @vzoBuh4l].

### Word Embeddings
Word embeddings are representations that map individual words to real valued vectors of user-specified dimensions.
These embeddings have been shown to capture the semantic and syntatic information between words [@u5iJzbp9].
Using all candidate sentences for each individual relationship pair, we trained facebook's fastText [@qUpCDz2v] to generate word embeddings.
The fastText model uses a skipgram model [@1GhHIDxuW] that aims to predict the context given a candidate word and pairs the model with a novel scoring function that treats each word as a bag of character n-grams.
We trained this model for 20 epochs using a window size of 2 and generated 300-dimensional word embeddings.
We use the optimized word embeddings to train a discriminative model.  

### Discriminator Model
talk about the discriminator model and how it works
### Discriminator Model Calibration
talk about calibrating deep learning models with temperature smoothing

### Experimental Design
Being able to re-use label functions across edge types would substantially reduce the number of label functions required to extract multiple relationship types from biomedical literature.
We first established a baseline by training a generative model using only distant supervision label functions designed for the target edge type.
As an example, for the gene-interacts-gene edge type we used label functions that returned a `1` if the pair of genes were included in the Human Interaction database [@LCyCrr7W], the iRefIndex database [@gtV3bOpd] or in the Incomplete Interactome database [@2jkcXYxN].
Then we compared models that also included text and domain-heuristic label functions.
Using a sampling with replacement approach, we sampled these text and domain-heuristic label functions separately within edge types, across edge types, and from a pool of all label functions.
We compared within-edge-type performance to across-edge-type and all-edge-type performance.
For each edge type we sampled a fixed number of label functions consisting of five evenly-spaced numbers between one and the total number of possible label functions.
We repeated this sampling process 50 times for each point.
We evaluated both generative and discriminative models at each point, and we report performance of each in terms of the area under the receiver operating characteristic curve (AUROC) and the area under the precision-recall curve (AUPR).


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

---
author-meta:
- David N. Nicholson
- Daniel S. Himmelstein
- Casey S. Greene
date-meta: '2019-07-08'
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
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/5ae21d8ab62aeac5995692dc0838e9785b105f1f/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@5ae21d8](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/5ae21d8ab62aeac5995692dc0838e9785b105f1f)
on July 8, 2019.
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


#Introduction
Set introduction for paper here
Talk about problem, goal, and significance of paper

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
We hand labeled five hundred to a thousand candidate sentences of each relationship to obtain to obtain a ground truth set (Table {@tbl:candidate-sentences}, [dataset](http://github.com/text_minded_hetnet_manuscript/master/supplementary_materials/annotated_sentences).

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
---
author-meta:
- David N. Nicholson
- Daniel S. Himmelstein
- Casey S. Greene
date-meta: '2020-01-23'
header-includes: '<!--

  Manubot generated metadata rendered from header-includes-template.html.

  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html

  -->

  <meta name="dc.format" content="text/html" />

  <meta name="dc.title" content="Reusing label functions to extract multiple types of relationships from biomedical abstracts at scale" />

  <meta name="citation_title" content="Reusing label functions to extract multiple types of relationships from biomedical abstracts at scale" />

  <meta property="og:title" content="Reusing label functions to extract multiple types of relationships from biomedical abstracts at scale" />

  <meta property="twitter:title" content="Reusing label functions to extract multiple types of relationships from biomedical abstracts at scale" />

  <meta name="dc.date" content="2020-01-23" />

  <meta name="citation_publication_date" content="2020-01-23" />

  <meta name="dc.language" content="en-US" />

  <meta name="citation_language" content="en-US" />

  <meta name="dc.relation.ispartof" content="Manubot" />

  <meta name="dc.publisher" content="Manubot" />

  <meta name="citation_journal_title" content="Manubot" />

  <meta name="citation_technical_report_institution" content="Manubot" />

  <meta name="citation_author" content="David N. Nicholson" />

  <meta name="citation_author_institution" content="Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania" />

  <meta name="citation_author_orcid" content="0000-0003-0002-5761" />

  <meta name="twitter:creator" content="@None" />

  <meta name="citation_author" content="Daniel S. Himmelstein" />

  <meta name="citation_author_institution" content="Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania" />

  <meta name="citation_author_orcid" content="0000-0002-3012-7446" />

  <meta name="twitter:creator" content="@dhimmel" />

  <meta name="citation_author" content="Casey S. Greene" />

  <meta name="citation_author_institution" content="Department of Systems Pharmacology and Translational Therapeutics, University of Pennsylvania" />

  <meta name="citation_author_orcid" content="0000-0001-8713-9213" />

  <meta name="twitter:creator" content="@GreeneScientist" />

  <link rel="canonical" href="https://greenelab.github.io/text_mined_hetnet_manuscript/" />

  <meta property="og:url" content="https://greenelab.github.io/text_mined_hetnet_manuscript/" />

  <meta property="twitter:url" content="https://greenelab.github.io/text_mined_hetnet_manuscript/" />

  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/text_mined_hetnet_manuscript/" />

  <meta name="citation_pdf_url" content="https://greenelab.github.io/text_mined_hetnet_manuscript/manuscript.pdf" />

  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/text_mined_hetnet_manuscript/manuscript.pdf" />

  <link rel="alternate" type="text/html" href="https://greenelab.github.io/text_mined_hetnet_manuscript/v/fe91de6685a61fc5d3b9e7936dda8c3146788425/" />

  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/text_mined_hetnet_manuscript/v/fe91de6685a61fc5d3b9e7936dda8c3146788425/" />

  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/text_mined_hetnet_manuscript/v/fe91de6685a61fc5d3b9e7936dda8c3146788425/manuscript.pdf" />

  <meta property="og:type" content="article" />

  <meta property="twitter:card" content="summary_large_image" />

  <meta property="og:image" content="https://github.com/greenelab/text_mined_hetnet_manuscript/raw/fe91de6685a61fc5d3b9e7936dda8c3146788425/thumbnail.png" />

  <meta property="twitter:image" content="https://github.com/greenelab/text_mined_hetnet_manuscript/raw/fe91de6685a61fc5d3b9e7936dda8c3146788425/thumbnail.png" />

  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />

  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />

  <meta name="theme-color" content="#ad1457" />

  <!-- end Manubot generated metadata -->'
keywords:
- machine learning
- weak supervision
- natural language processing
- heterogenous netowrks
- text mining
lang: en-US
title: Reusing label functions to extract multiple types of relationships from biomedical abstracts at scale
...



_A DOI-citable version of this manuscript is available at <https://doi.org/10.1101/730085>_.


<small><em>
This manuscript
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/fe91de6685a61fc5d3b9e7936dda8c3146788425/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@fe91de6](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/fe91de6685a61fc5d3b9e7936dda8c3146788425)
on January 23, 2020.
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

Knowledge bases support multiple research efforts such as providing contextual information for biomedical entities, constructing networks, and supporting the interpretation of high-throughput analyses.
Some knowledge bases are automatically constructed, but most are populated via some form of manual curation.
Manual curation is time consuming and difficult to scale in the context of an increasing publication rate.
A recently described "data programming" paradigm seeks to circumvent this arduous process by combining distant supervision with simple rules and heuristics written as labeling functions that can be automatically applied to inputs.
Unfortunately writing useful label functions requires substantial error analysis and is a nontrivial task: in early efforts to use data programming we found that producing each label function could take a few days.
Producing a biomedical knowledge base with multiple node and edge types could take hundreds or possibly thousands of label functions.
In this paper we sought to evaluate the extent to which label functions could be re-used across edge types. 
We used a subset of Hetionet v1 that centered on disease, compound, and gene nodes to evaluate this approach.
We compared a baseline distant supervision model with the same distant supervision resources added to edge-type-specific label functions, edge-type-mismatch label functions, and all label functions.
We confirmed that adding additional edge-type-specific label functions improves performance.
We also found that adding one or a few edge-type-mismatch label functions nearly always improved performance.
Adding a large number of edge-type-mismatch label functions produce variable performance that depends on the edge type being predicted and the label function's edge type source.
Lastly, we show that this approach, even on this subgraph of Hetionet, could add new edges to Hetionet v1 with high confidence.
We expect that practical use of this strategy would include additional filtering and scoring methods which would further enhance precision.


## Introduction

Knowledge bases are important resources that hold complex structured and unstructed information. 
These resources have been used in important tasks such as network analysis for drug repurposing discovery [@u8pIAt5j; @bPvC638e; @O21tn8vf] or as a source of training labels for text mining systems [@EHeTvZht; @CVHSURuI; @HS4ARwmZ]. 
Populating knowledge bases often requires highly-trained scientists to read biomedical literature and summarize the results [@N1Ai0gaI].
This manual curation process requires a significant amount of effort and time: in 2007 researchers estimated that filling in the missing annotations would require approximately 8.4 years [@UdzvLgBM].
The rate of publications has continued to increase exponentially [@1DBISRlwN].
This has been recognized as a considerable challenge, which can lead to gaps in knowledge bases [@UdzvLgBM].  
Relationship extraction has been studied as a solution towards handling this problem [@N1Ai0gaI].
This process consists of creating a machine learning system to automatically scan and extract relationships from textual sources.
Machine learning methods often leverage a large corpus of well-labeled training data, which still requires manual curation.
Distant supervision is one technique to sidestep the requirement of well-annotated sentences: with distant supervision one makes the assumption that all sentences containing an entity pair found in a selected database provide evidence for a relationship [@EHeTvZht].
Distant supervision provides many labeled examples; however it is accompanied by a decrease in the quality of the labels.  
Ratner et al. [@5Il3kN32] recently introduced "data programming" as a solution.
Data programming combines distant supervision with the automated labeling of text using hand-written label functions.
The distant supervision sources and label functions are integrated using a noise aware generative model that is used to produce training labels.
Combining distant supervision with label functions can dramatically reduce the time required to acquire sufficient training data.
However, constructing a knowledge base of heterogeneous relationships through this framework still requires tens of hand-written label functions for each relationship type.
Writing useful label functions requires significant error analysis, which can be a time-consuming process.  

In this paper, we aim to address the question: to what extent can label functions be re-used across different relationship types?
We hypothesized that sentences describing one relationship type may share information in the form of keywords or sentence structure with sentences that indicate other relationship types.
We designed a series of experiments to determine the extent to which label function re-use enhanced performance over distant supervision alone.
We examined relationships that indicated similar types of physical interactions (i.e., gene-binds-gene and compound-binds-gene) as well as different types (i.e., disease-associates-gene and compound-treats-disease).
The re-use of label functions could dramatically reduce the number required to generate and update a heterogeneous knowledge graph.

### Related Work

Relationship extraction is the process of detecting and classifying semantic relationships from a collection of text.
This process can be broken down into three different categories: (1) the use of natural language processing techniques such as manually crafted rules and the identification of key text patterns for relationship extraction, (2) the use of unsupervised methods via co-occurrence scores or clustering, and (3) supervised or semi-supervised machine learning using annotated datasets for the classification of documents or sentences.
In this section, we discuss selected efforts for each type of edge that we include in this project.

#### Disease-Gene Associations 

Efforts to extract Disease-associates-Gene (DaG) relationships have often used manually crafted rules or unsupervised methods.
One study used hand crafted rules based on a sentence's grammatical structure, represented as dependency trees, to extract DaG relationships [@NLxmpSdj].
Some of these rules inspired certain DaG text pattern label functions in our work.
Another study used co-occurrence frequencies within abstracts and sentences to score the likelihood of association between disease and gene pairs [@5gG8hwv7].
The results of this study were incorporated into Hetionet v1 [@O21tn8vf], so this served as one of our distant supervision label functions.
Another approach built off of the above work by incorporating a supervised classifier, trained via distant supervision, into a scoring scheme [@IGXdryzB].
Each sentence containing a disease and gene mention is scored using a logistic regression model and combined using the same co-occurrence approach used in Pletscher-Frankild et al. [@5gG8hwv7].
We compared our results to this approach to measure how well our overall method performs relative to other methods.
Besides the mentioned three studies, researchers have used co-occurrences for extraction alone [@19zkt9R1G; @WDNuFZ4j; @DGlWGDEt] or in combination with other features to recover DaG relationships [@CxErbNTp].
One recent effort relied on a bi-clustering approach to detect DaG-relevant sentences from Pubmed abstracts [@CSiMoOrI] with clustering of dependency paths grouping similar sentences together.
The results of this work supply our domain heuristic label functions.
These approaches do not rely on a well-annotated training performance and tend to provide excellent recall, though the precision is often worse than with supervised methods [@199TFjkrC; @1ZjlFRHa].

Hand-crafted high-quality datasets [@hbAqN08A; @Y2DcwTrA; @luGt8luc; @1Du6MinB8] often serve as a gold standard for training, tuning, and testing supervised machine learning methods in this setting.
Support vector machines have been repeatedly used to detect DaG relationships [@hbAqN08A; @3j1T67vB; @GeCe9qfW].
These models perform well in large feature spaces, but are slow to train as the number of data points becomes large.
Recently, some studies have used deep neural network models.
One used a pre-trained recurrent neural network [@riimmjYr], and another used distant supervision [@k7ZUI6FL].
Due to the success of these two models, we decided to use a deep neural network as our discriminative model.

#### Compound Treats Disease

The goal of extracting Compound-treats-Disease (CtD) edges is to identify sentences that mention current drug treatments or propose new uses for existing drugs.
One study combined an inference model from previously established drug-gene and gene-disease relationships to infer novel drug-disease interactions via co-occurrences [@ETC6lm7S].
A similar approach has also been applied to CtD extraction [@AdKPf5EO].
Manually-curated rules have also been applied to PubMed abstracts to address this task [@1avvFjJ9].
The rules were based on identifying key phrases and wordings related to using drugs to treat a disease, and we used these patterns as inspirations for some of our CtD label functions. 
Lastly, one study used a  bi-clustering approach to identify sentences relevant to CtD edges [@CSiMoOrI].
As with DaG edges, we use the results from this study to provide what we term as domain heuristic label functions.

Recent work with supervised machine learning methods has often focused on compounds that induce a disease: an important question for toxicology and the subject of the BioCreative V dataset [@6wNuLZWb].
We don't consider environmental toxicants in our work, as our source databases for distant supervision are primarily centered around FDA-approved therapies.

#### Compound Binds Gene

The BioCreative VI track 5 task focused on classifying compound-protein interactions and has led to a great deal of work on the topic [@16As8893j].
The equivalent edge in our networks is Compound-binds-Gene (CbG).
Curators manually annotated 2,432 PubMed abstracts for five different compound protein interactions (agonist, antagonist, inhibitor, activator and substrate/product production) as part of the BioCreative task. 
The best performers on this task achieved an F1 score of 64.10% [@16As8893j].
Numerous additional groups have now used the publicly available dataset, that resulted from this competition, to train supervised machine learning methods [@OnvaFHG9; @i7KpvzCo; @5LOkzCNK; @riimmjYr; @5LOkzCNK; @1H34cFSl8; @16MGWGDUB; @1HjIKY59u; @WP5p3RT3] and semi-supervised machine learning methods [@P2pnebCX].
These approaches depend on well-annotated training datasets, which creates a bottleneck.
In addition to supervised and semi-supervised machine learning methods, hand crafted rules [@107WYOcxW] and bi-clustering of dependency trees  [@CSiMoOrI] have been used.
We use the results from the bi-clustering study to provide a subset of the CbG label functions in this work.

#### Gene-Gene Interactions

Akin to the DaG edge type, many efforts to extract Gene-interacts-Gene (GiG) relationships used co-occurrence approaches.
This edge type is more frequently referred to as a protein-protein interaction.
Even approaches as simple as calculating Z-scores from PubMed abstract co-occurrences can be informative [@q9Fhy8eq], and there are numerous studies using co-occurrences [@yGMDz6lK; @w32u0Rj9; @8GVs1dBG; @DGlWGDEt].
However, more sophisticated strategies such as distant supervision appear to improve performance [@IGXdryzB].
Similarly to the other edge types, the bi-clustering approach over dependency trees has also been applied to this edge type [@CSiMoOrI].
This manuscript provides a set of label functions for our work.

Most supervised classifiers used publicly available datasets for evaluation [@YWh6tPj; @DWpAeBxB; @szMMEMdC; @L9IIm3Zd; @115pgEuOr].
These datasets are used equally among studies, but can generate noticeable  differences in terms of performance [@DR8XM4Ff].
Support vector machines were a common approach to extract GiG edges [@iiQkIqUX; @1B0lnkj35].
However, with the growing popularity of deep learning numerous deep neural network architectures have been applied [@ibJfUvEe; @1H4LpFrU0; @bLKJwjMD; @P2pnebCX].
Distant supervision has also been used in this domain [@WYud0jQT], and in fact this effort was one of the motivating rationales for our work.


<style> 
span.gene_color { color:#02b3e4 } 
span.disease_color { color:#875442 } 
span.compound_color { color:#e91e63 }
 </style> 

## Materials and Methods

### Hetionet

![
A metagraph (schema) of Hetionet where biomedical entities are represented as nodes and the relationships between them are represented as edges.
We examined performance on the highlighted subgraph; however, the long-term vision is to capture edges for the entire graph.
](images/figures/hetionet/metagraph_highlighted_edges.png){#fig:hetionet}

Hetionet [@O21tn8vf] is a large heterogenous network that contains pharmacological and biological information.
This network depicts information in the form of nodes and edges of different types: nodes that represent biological and pharmacological entities and edges which represent relationships between entities. 
Hetionet v1.0 contains 47,031 nodes with 11 different data types and 2,250,197 edges that represent 24 different relationship types (Figure {@fig:hetionet}).
Edges in Hetionet were obtained from open databases, such as the GWAS Catalog [@16cIDAXhG] and DrugBank [@1FI8iuYiQ].
For this project, we analyzed performance over a subset of the Hetionet relationship types: disease associates with a gene (DaG), compound binds to a gene (CbG), gene interacts with gene (GiG) and compound treating a disease (CtD).

### Dataset

We used PubTator [@13vw5RIy4] as input to our analysis.
PubTator provides MEDLINE abstracts that have been annotated with well-established entity recognition tools including DNorm [@vtuZ3Wx7] for disease mentions, GeneTUKit [@4S2HMNpa] for gene mentions, Gnorm [@1AkC7QdyP] for gene normalizations and a dictionary based search system for compound mentions [@r501gnuM].
We downloaded PubTator on June 30, 2017, at which point it contained 10,775,748 abstracts. 
Then we filtered out mention tags that were not contained in hetionet.
We used the Stanford CoreNLP parser [@RQkLuc5t] to tag parts of speech and generate dependency trees.
We extracted sentences with two or more mentions, termed candidate sentences.
Each candidate sentence was stratified by co-mention pair to produce a training set, tuning set and a testing set (shown in Table {@tbl:candidate-sentences}).
Each unique co-mention pair is sorted into four categories: (1) in hetionet and has sentences, (2) in hetionet and doesn't have sentences, (3) not in hetionet and does have sentences and (4) not in hetionet and doesn't have sentences.
Within these four categories each pair is randomly assigned their own individual partition rank (continuous number between 0 and 1).
Any rank lower than 0.7 is sorted into the training set, while any rank greater than 0.7 and lower than 0.9 is assigned to the tuning set.
The rest of the pairs with a rank greater than or equal to 0.9 is assigned to the test set.
Sentences that contain more than one co-mention pair are treated as multiple individual candidates.
We hand labeled five hundred to a thousand candidate sentences of each relationship type to obtain a ground truth set (Table {@tbl:candidate-sentences})[^1].

[^1]: Labeled sentences are available [here](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/master/supplementary_materials/annotated_sentences).

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

### Label Functions for Annotating Sentences

The challenge of having too few ground truth annotations is common to many natural language processing settings, even when unannotated text is abundant.
Data programming circumvents this issue by quickly annotating large datasets by using multiple noisy signals emitted by label functions [@5Il3kN32].
Label functions are simple pythonic functions that emit: a positive label (1), a negative label (-1) or abstain from emitting a label (0).
We combine these functions using a generative model to output a single annotation, which is a consensus probability score bounded between 0 (low chance of mentioning a relationship) and 1 (high chance of mentioning a relationship).
We used these annotations to train a discriminator model that makes the final classification step.
Our label functions fall into three categories: databases, text patterns and domain heuristics.
We provide examples for each category in our [supplemental methods section](#label-function-categories).  

### Training Models

#### Generative Model

The generative model is a core part of this automatic annotation framework.
It integrates multiple signals emitted by label functions and assigns a training class to each candidate sentence.
This model assigns training classes by estimating the joint probability distribution of the latent true class ($Y$) and label function signals ($\Lambda$), ($P_{\theta}(\Lambda, Y)$).
Assuming each label function is conditionally independent, the joint distribution is defined as follows:  

$$
P_{\theta}(\Lambda, Y) = \frac{\exp(\sum_{i=1}^{m} \theta^{T}F_{i}(\Lambda, y))}
{\sum_{\Lambda'}\sum_{y'} \exp(\sum_{i=1}^{m} \theta^{T}F_{i}(\Lambda', y'))}
$$  

where $m$ is the number of candidate sentences, $F$ is the vector of summary statistics and $\theta$ is a vector of weights for each summary statistic.
The summary statistics used by the generative model are as follows:  

$$F^{Lab}_{i,j}(\Lambda, Y) = \unicode{x1D7D9}\{\Lambda_{i,j} \neq 0\}$$
$$F^{Acc}_{i,j}(\Lambda, Y) = \unicode{x1D7D9}\{\Lambda_{i,j} = y_{i,j}\}$$   

*Lab* is the label function's propensity (the frequency of a label function emitting a signal).
*Acc* is the individual label function's accuracy given the training class.
This model optimizes the weights ($\theta$) by minimizing the negative log likelihood:

$$\hat{\theta} = argmin_{\theta} -\sum_{\Lambda} \sum_{Y} log P_{\theta}(\Lambda, Y)$$

In the framework we used predictions from the generative model, $\hat{Y} = P_{\hat{\theta}}(Y \mid \Lambda)$, as training classes for our dataset [@vzoBuh4l; @9Jo1af7Z]. 

### Experimental Design

Being able to re-use label functions across edge types would substantially reduce the number of label functions required to extract multiple relationships from biomedical literature.
We first established a baseline by training a generative model using only distant supervision label functions designed for the target edge type.
For example, in the Gene interacts Gene (GiG) edge type we used label functions that returned a **1** if the pair of genes were included in the Human Interaction database [@LCyCrr7W], the iRefIndex database [@gtV3bOpd] or in the Incomplete Interactome database [@2jkcXYxN].
Then we compared the baseline model with models that also included text and domain-heuristic label functions.
Using a sampling with replacement approach, we sampled these text and domain-heuristic label functions separately within edge types, across edge types, and from a pool of all label functions.
We compared within-edge-type performance to across-edge-type and all-edge-type performance.
For each edge type we sampled a fixed number of label functions consisting of five evenly spaced numbers between one and the total number of possible label functions.
We repeated this sampling process 50 times for each point.
We evaluated both generative and discriminative (training and downstream analyses are described in the [supplemental methods section](#discriminative-model)) models at each point, and report performance of each in terms of the area under the receiver operating characteristic curve (AUROC) and the area under the precision-recall curve (AUPR).


## Results

### Generative Model Using Randomly Sampled Label Functions
![
Grid of AUROC scores for each generative model trained on randomly sampled label functions.
The rows depict the relationship each model is trying to predict and the columns are the edge type specific sources from which each label function is sampled.
The right most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/label_sampling_experiment/transfer_test_set_auroc.png){#fig:auroc_gen_model_performance}

We added randomly sampled label functions to a baseline for each edge type to evaluate the feasibility of label function re-use.
Our baseline model consisted of a generative model trained with only edge-specific distant supervision label functions.
We reported the results in AUROC and AUPR (Figure {@fig:auroc_gen_model_performance} and Supplemental Figure {@fig:aupr_gen_model_performance}).  
The on-diagonal plots of figure {@fig:auroc_gen_model_performance} and supplemental figure {@fig:aupr_gen_model_performance} show increasing performance when edge-specific label functions are added on top of the  edge-specific baselines.
The CtD edge type is a quintessential example of this trend.
The baseline model starts off with an AUROC score of 52% and an AUPRC of 28%, which increase to 76% and 49% respectively as more CtD label functions are included. 
DaG edges have a similar trend: performance starting off with an AUROC of 56% and AUPR of 41% then increases to 62% and 45% respectively.
Both the CbG and GiG edges have an increasing trend but plateau after a few label functions are added.  

The off-diagonals in figure {@fig:auroc_gen_model_performance} and supplemental figure {@fig:aupr_gen_model_performance} show how performance varies when label functions from one edge type are added to a different edge type's baseline.
In certain cases (apparent for DaG), performance increases regardless of the edge type used for label functions.
In other cases (apparent with CtD), one label function appears to improve performance; however, adding more label functions does not improve performance (AUROC) or decreases it (AUPR).
In certain cases, the source of the label functions appears to be important: the performance of CbG edges decrease when using label functions from the DaG and CtD categories.

Our initial hypothesis was based on the idea that certain edge types capture similar physical relationships and that these cases would be particularly amenable for label function transfer.
For example, CbG and GiG both describe physical interactions.
We observed that performance increased as assessed by both AUROC and AUPR when using label functions from the GiG edge type to predict CbG edges.
A similar trend was observed when predicting the GiG edge; however, the performance differences were small for this edge type making the importance difficult to assess.  
The last column shows increasing performance (AUROC and AUPR) for both DaG and CtD when sampling from all label functions.
CbG and GiG also had increased performance when one random label function was sampled, but performance decreased drastically as more label functions were added.
It is possible that a small number of irrelevant label functions are able to overwhelm the distant supervision label functions in these cases (see Figure {@fig:auroc_random_label_function_performance} and Supplemental Figure {@fig:aupr_random_label_function_performance}).

### Random Label Function Generative Model Analysis
![
A grid of AUROC (A) scores for each edge type.
Each plot consists of adding a single label function on top of the baseline model.
This label function emits a positive (shown in blue) or negative (shown in orange) label at specified frequencies, and performance at zero is equivalent to not having a randomly emitting label function.
The error bars represent 95% confidence intervals for AUROC or AUPR (y-axis) at each emission frequency.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/gen_model_error_analysis/transfer_test_set_auroc.png){#fig:auroc_random_label_function_performance}

We observed that including one label function of a mismatched type to distant supervision often improved performance, so we evaluated the effects of adding a random label function in the same setting.
We found that usually adding random noise did not improve performance (Figure {@fig:auroc_random_label_function_performance} and Supplemental Figure {@fig:aupr_random_label_function_performance}).
For the CbG edge type we did observe slightly increased performance via AUPR (Supplemental Figure {@fig:aupr_random_label_function_performance}).
However, performance changes in general were smaller than those observed with mismatched label types.

### Discriminative Model Calibration

![
Deep learning models are overconfident in their predictions and need to be calibrated after training.
These are calibration plots for the discrimintative model.
The green line represents the predictions before calibration and the blue line shows predictions after calibration. 
Data points that lie closer to diagonal line show better model calibration, while data points far from the diagonal show poor performance.
A perfectly calibrated model would align straight along the diagonal line. 
](https://raw.githubusercontent.com/danich1/snorkeling/86037d185a299a1f6dd4dd68605073849c72af6f/figures/model_calibration_experiment/model_calibration.png){#fig:discriminative_model_calibration}

Even deep learning models with good AUROC and AUPR statistics can be subject to poor calibration.
Typically, these models are overconfident in their predictions [@QJ6hYH8N; @rLVjMJ5l].
We attempted to use temperature scaling to fix the calibration of the best performing discriminative models (Figure {@fig:discriminative_model_calibration}).
Before calibration (green lines), our models were aligned with the ideal calibration only when predicting low probability scores (close to 0.25).
Applying the temperature scaling calibration algorithm (blue lines) did not substantially improve the calibration of the model in most cases.
The exception to this pattern is the Disease associates Gene (DaG) model where high confidence scores are shown to be better calibrated.
Overall, calbrating deep learning models is a nontrivial task that requires  more complex approaches to accomplish.


## Discussion

We tested the feasibility of re-using label functions to extract relationships from literature.
Through our sampling experiment, we found that adding relevant label functions increases prediction performance (shown in the on-diagonals of Figures {@fig:auroc_gen_model_performance} and Supplemental Figure {@fig:aupr_gen_model_performance}).
We found that label functions designed from relatively related edge types can increase performance (seen when GiG label functions predicts CbG and vice versa).
We noticed that one edge type (DaG) is agnostic to label function source (Figure {@fig:auroc_gen_model_performance} and Supplemental Figure {@fig:aupr_gen_model_performance}). 
Performance routinely increases when adding a single mismatched label function to our baseline model (the generative model trained only on distant supervision label functions).
These results led us to hypothesize that adding a small amount of noise aided the model, but our experiment with a random label function reveals that this was not the case (Figures {@fig:auroc_random_label_function_performance} and {@fig:aupr_random_label_function_performance}).
Based on these results one question still remains: why does performance drastically increase when adding a single label function to our distant supervision baseline?

The discriminative model didn't work as intended. 
The majority of the time the discriminative model underperformed the generative model (Supplemental Figures {@fig:auroc_discriminative_model_performance} and {@fig:aupr_discriminative_model_performance}).
Potential reasons for this are the discriminative model overfitting to the generative model's predictions and a negative class bias in some of our datasets (Table {@tbl:candidate-sentences}).
The challenges with the discriminative model are likely to have led to issues in our downstream analyses: poor model calibration (Supplemental Figure {@fig:discriminative_model_calibration}) and poor recall in detecting existing Hetionet edges (Supplemental Figure {@fig:hetionet_reconstruction}).
Despite the above complications, our model had similar performance with a published baseline model (Supplemental Figure {@fig:cocoscore_comparison}).
This implies that with better tuning the discriminative model has the potential to perform better than the baseline model.


## Conclusion and Future Direction

Filling out knowledge bases via manual curation can be an arduous and erroneous task [@UdzvLgBM].
As the rate of publications increases manual curation becomes an infeasible approach.
Data programming, a paradigm that uses label functions as a means to speed up the annotation process, can be used as a solution for this problem.
A problem with this paradigm is that creating a useful label function takes a significant amount of time. 
We tested the feasibility of reusing label functions as a way to speed up the  label function creation process.
We conclude that label function re-use across edge types can increase performance when there are certain constraints on the number of functions re-used.
More sophisticated methods of reuse may be able to capture many of the advantages and avoid many of the drawbacks.
Adding more relevant label functions can increase overall performance.
The discriminative model, under this paradigm, has a tendency to overfit to predictions of the generative model.
We recommend implementing regularization techniques such as drop out and weight decay to combat this issue.

This work sets up the foundation for creating a common framework that mines text to create edges.
Within this framework we would continuously ingest new knowledge as novel findings are published, while providing a single confidence score for an edge by consolidating sentence scores.
Different from existing hetnets like Hetionet where text-derived edges generally cannot be exactly attributed to excerpts from literature [@O21tn8vf; @L2B5V7XC], our approach would annotate each edge with its source sentences.
In addition, edges generated with this approach would be unencumbered from upstream licensing or copyright restrictions, enabling openly licensed hetnets at a scale not previously possible [@4G0GW8oe; @137tbemL9; @1GwdMLPbV].
Accordingly, we plan to use this framework to create a robust multi-edge extractor via multitask learning [@9Jo1af7Z] to construct continuously updating literature-derived hetnets.


## Supplemental Information

This manuscript and supplemental information are available at <https://greenelab.github.io/text_mined_hetnet_manuscript/>.
Source code for this work is available under open licenses at: <https://github.com/greenelab/snorkeling/>.

## Acknowledgements

The authors would like to thank Christopher Ré's group at Stanford Univeristy, especially Alex Ratner and Steven Bach, for their assistance with this project.
We also want to thank Graciela Gonzalez-Hernandez for her advice and input with this project.
This work was support by [Grant GBMF4552](https://www.moore.org/grant-detail?grantId=GBMF4552) from the Gordon Betty Moore Foundation.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>

## Supplemental Methods {.page_break_before}

### Label Function Categories

We provide examples of label function categories below. Each example regards the following candidate sentence: “[PTK6]{.gene_color} may be a novel therapeutic target for [pancreatic cancer]{.disease_color}.”

**Databases**: These label functions incorporate existing databases to generate a signal, as seen in distant supervision [@EHeTvZht].
These functions detect if a candidate sentence's co-mention pair is present in a given database.
If the candidate pair is present, our label function emitted a positive label and abstained otherwise.
If the candidate pair wasn't present in any existing database, a separate label function emitted a negative label.
We used a separate label function to prevent a label imbalance problem that we encountered during development: emitting positive and negatives from the same label functions appeared to result in classifiers that predict almost exclusively negative predictions.

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
For this category, we used dependency path cluster themes generated by Percha et al. [@CSiMoOrI].
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
| DaG | 7 | 20 | 10 | 
| CtD | 3 | 15 | 7 |
| CbG | 9 | 13 | 7 | 
| GiG | 9 | 20 | 8 | 

Table: The distribution of each label function per relationship. {#tbl:label-functions} 

### Adding Random Noise to Generative Model

We discovered in the course of this work that adding a single label function from a mismatched type would often improve the performance of the generative model (see Results).
We designed an experiment to test whether adding a noisy label function also increased performance.
This label function emitted a positive or negative label at varying frequencies, which were evenly spaced from zero to one.
Zero was the same as distant supervision and one meant that all sentences were randomly labeled.
We trained the generative model with these label functions added and reported results in terms of AUROC and AUPR.

### Discriminative Model

The discriminative model is a neural network, which we train to predict labels from the generative model.
The expectation is that the discriminative model can learn more complete features of the text than the label functions used in the generative model.
We used a convolutional neural network with multiple filters as our discriminative model.
This network uses multiple filters with fixed widths of 300 dimensions and a fixed height of 7 (Figure {@fig:convolutional_network}), because this height provided the best performance in terms of relationship classification [@fs8rAHoJ].
We trained this model for 20 epochs using the adam optimizer [@c6d3lKFX] with pytorch's default parameter settings and a learning rate of 0.001.
We added a L2 penalty on the network weights to prevent overfitting.
Lastly, we added a dropout layer (p=0.25) between the fully connected layer and the softmax layer.

![
The architecture of the discriminative model was a convolutional neural network.
We performed a convolution step using multiple filters. 
The filters generated a feature map that was sent into a maximum pooling layer that was designed to extract the largest feature in each map.
The extracted features were concatenated into a singular vector that was passed into a fully connected network. 
The fully connected network had 300 neurons for the first layer, 100 neurons for the second layer and 50 neurons for the last layer. 
The last step from the fully connected network was to generate predictions using a softmax layer.
](images/figures/convolutional_neural_network/convolutional_neural_nework.png){#fig:convolutional_network}

#### Word Embeddings

Word embeddings are representations that map individual words to real valued vectors of user-specified dimensions.
These embeddings have been shown to capture the semantic and syntactic information between words [@u5iJzbp9].
We trained Facebook's fastText [@qUpCDz2v] using all candidate sentences for each individual relationship pair to generate word embeddings.
fastText uses a skipgram model [@1GhHIDxuW] that aims to predict the surrounding context for a candidate word and pairs the model with a novel scoring function that treats each word as a bag of character n-grams.
We trained this model for 20 epochs using a window size of 2 and generated 300-dimensional word embeddings.
We use the optimized word embeddings to train a discriminative model. 

#### Calibration of the Discriminative Model

Often many tasks require a machine learning model to output reliable probability predictions. 
A model is well calibrated if the probabilities emitted from the model match the observed probabilities: a well-calibrated model that assigns a class label with 80% probability should have that class appear 80% of the time.
Deep neural network models can often be poorly calibrated [@QJ6hYH8N; @rLVjMJ5l].
These models are usually over-confident in their predictions.
As a result, we calibrated our convolutional neural network using temperature scaling. 
Temperature scaling uses a parameter T to scale each value of the logit vector (z) before being passed into the softmax (SM) function.

$$\sigma_{SM}(\frac{z_{i}}{T}) = \frac{\exp(\frac{z_{i}}{T})}{\sum_{i}\exp(\frac{z_{i}}{T})}$$

We found the optimal T by minimizing the negative log likelihood (NLL) of a held out validation set.
The benefit of using this method is that the model becomes more reliable and the accuracy of the model doesn't change [@QJ6hYH8N].

## Supplemental Tables and Figures

### Generative Model AUPR Performance

![
Grid of AUPR scores for each generative model trained on randomly sampled label functions.
The rows depict the relationship each model is trying to predict and the columns are the edge type specific sources from which each label function is sampled.
For example, the top-left most square depicts the generative model predicting DaG sentences, while randomly sampling label functions designed to predict the DaG relationship. 
The square towards the right depicts the generative model predicting DaG sentences, while randomly sampling label functions designed to predict the CtD relationship.
This pattern continues filling out the rest of the grid.
The right most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/label_sampling_experiment/transfer_test_set_auprc.png){#fig:aupr_gen_model_performance}

### Random Label Function Generative Model Analysis
![
A grid of AUROC (A) scores for each edge type.
Each plot consists of adding a single label function on top of the baseline model.
This label function emits a positive (shown in blue) or negative (shown in orange) label at specified frequencies, and performance at zero is equivalent to not having a randomly emitting label function.
The error bars represent 95% confidence intervals for AUROC or AUPR (y-axis) at each emission frequency.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/gen_model_error_analysis/transfer_test_set_auprc.png){#fig:aupr_random_label_function_performance}

### Discriminative Model Performance

![
Grid of AUROC scores for each discriminative model trained using generated labels from the generative models.
The rows depict the edge type each model is trying to predict and the columns are the edge type specific sources from which each label function was sampled. 
For example, the top-left most square depicts the discriminator model predicting DaG sentences, while randomly sampling label functions designed to predict the DaG relationship.
The error bars over the points represents the standard deviation between sampled runs.
The square towards the right depicts the discriminative model predicting DaG sentences, while randomly sampling label functions designed to predict the CtD relationship.
This pattern continues filling out the rest of the grid.
The right most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/label_sampling_experiment/disc_performance_test_set_auroc.png){#fig:auroc_discriminative_model_performance}

In this framework we used a generative model trained over label functions to produce probabilistic training labels for each sentence.
Then we trained a discriminative model, which has full access to a representation of the text of the sentence, to predict the generated labels.
The discriminative model is a convolutional neural network trained over word embeddings (See Methods).
We report the results of the discriminative model using AUROC and AUPR (Figures {@fig:auroc_discriminative_model_performance} and {@fig:aupr_discriminative_model_performance}).  
  
We found that the discriminative model under-performed the generative model in most cases.
Only for the CtD edge does the discriminative model appear to provide performance above the generative model and that increased performance is only with a modest number of label functions.
With the full set of label functions, performance of both models remain similar.
The one or a few mismatched label functions (off-diagonal) improving generative model performance trend is retained despite the limited performance of the discriminative model.

![
Grid of AUPR scores for each discriminative model trained using generated labels from the generative models.
The rows depict the edge type each model is trying to predict and the columns are the edge type specific sources from which each label function was sampled. 
For example, the top-left most square depicts the discriminator model predicting DaG sentences, while randomly sampling label functions designed to predict the DaG relationship.
The error bars over the points represents the standard deviation between sampled runs.
The square towards the right depicts the discriminative model predicting DaG sentences, while randomly sampling label functions designed to predict the CtD relationship.
This pattern continues filling out the rest of the grid.
The right most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/danich1/snorkeling/ee638b4e45717a86f54a2744a813baaa90bc6b84/figures/label_sampling_experiment/disc_performance_test_set_auprc.png){#fig:aupr_discriminative_model_performance}

#### Model Calibration Tables

| Disease Name           | Gene Symbol | Text     | Before Calibration | After Calibration | 
|------------------------|-------------|---------------------------------------------------------|--------------------|-------------------| 
| prostate cancer        | DKK1        | conclusion : high [dkk-1]{.gene_color} serum levels are associated with a poor survival in patients with [prostate cancer]{.disease_color} . | 0.999              | 0.916             | 
| breast cancer          | ERBB2       | conclusion : [her-2 / neu]{.gene_color} overexpression in primary [breast carcinoma]{.disease_color} is correlated with patients ' age ( under age 50 ) and calcifications at mammography . | 0.998              | 0.906             | 
| breast cancer          | ERBB2       | the results of multiple linear regression analysis , with her2 as the dependent variable , showed that family history of [breast cancer]{.disease_color} was significantly associated with elevated [her2]{.gene_color} levels in the tumors ( p = 0.0038 ) , after controlling for the effects of age , tumor estrogen receptor , and dna index .  | 0.998              | 0.904             | 
| colon cancer           | SP3         | ba also decreased expression of sp1 , [sp3]{.gene_color} and sp4 transcription factors which are overexpressed in [colon cancer]{.disease_color} cells and decreased levels of several sp-regulated genes including survivin , vascular endothelial growth factor , p65 sub-unit of nfkb , epidermal growth factor receptor , cyclin d1 , and pituitary tumor transforming gene-1 . | 0.998              | 0.902             | 
| breast cancer          | ERBB2       | in [breast cancer]{.disease_color} , overexpression of [her2]{.gene_color} is associated with an aggressive tumor phenotype and poor prognosis . | 0.998              | 0.898             | 
| breast cancer          | BCL2        | in clinical [breast cancer]{.disease_color} samples , high [bcl2]{.gene_color} expression was associated with poor prognosis . | 0.997              | 0.886             | 
| adrenal gland cancer   | TP53        | the mechanisms of adrenal tumorigenesis remain poorly established ; the r337h germline mutation in the [p53]{.gene_color} gene has previously been associated with [acts]{.disease_color} in brazilian children . | 0.996              | 0.883             | 
| prostate cancer        | AR          | the [androgen receptor]{.gene_color} was expressed in all primary and metastatic [prostate cancer]{.disease_color} tissues and no mutations were identified .  | 0.996              | 0.881             | 
| urinary bladder cancer | PIK3CA      | conclusions : increased levels of fgfr3 and [pik3ca]{.gene_color} mutated dna in urine and plasma are indicative of later progression and metastasis in [bladder cancer]{.disease_color} . | 0.995              | 0.866             | 
| ovarian cancer         | EPAS1       | the log-rank test showed that nuclear positive immunostaining for hif-1alpha ( p = .002 ) and cytoplasmic positive immunostaining for [hif-2alpha]{.gene_color} ( p = .0112 ) in tumor cells are associated with poor prognosis of patients with [ovarian carcinoma]{.disease_color} . | 0.994              | 0.86              |   

Table: Contains the top ten Disease-associates-Gene confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Gene mentions are highlighted in [blue]{.gene_color}. {#tbl:dg_top_ten_table}

| Disease Name                   | Gene Symbol | Text    | Before Calibration | After Calibration | 
|--------------------------------|-------------|----------------------------------|--------------------|-------------------| 
| endogenous depression          | EP300       | from a clinical point of view , [p300]{.gene_color} amplitude should be considered as a psychophysiological index of suicidal risk in major [depressive disorder]{.disease_color} .  | 0.202              | 0.379             | 
| Alzheimer's disease            | PDK1        | [ from prion diseases to [alzheimer 's disease]{.disease_color} : a common therapeutic target , [pdk1 ]]{.gene_color} .  | 0.2                | 0.378             | 
| endogenous depression          | HTR1A       | gepirone , a selective serotonin ( [5ht1a )]{.gene_color} partial agonist in the treatment of [major depression]{.disease_color} . | 0.199              | 0.378             | 
| Gilles de la Tourette syndrome | FGF9        | there were no differences in gender distribution , age at tic onset or [td]{.disease_color} diagnosis , tic severity , proportion with current diagnoses of ocd/oc behavior or attention deficit hyperactivity disorder ( adhd ) , cbcl internalizing , externalizing , or total problems scores , ygtss scores , or [gaf]{.gene_color} scores . | 0.185              | 0.37              | 
| hematologic cancer             | MLANA       | methods : the sln sections ( n = 214 ) were assessed by qrt assay for 4 established messenger rna biomarkers : [mart-1]{.gene_color} , mage-a3 , [galnac-t]{.disease_color} , and pax3 . | 0.18               | 0.368             | 
| endogenous depression          | MAOA        | alpha 2-adrenoceptor responsivity in [depression]{.disease_color} : effect of chronic treatment with moclobemide , a selective [mao-a-inhibitor]{.gene_color} , versus maprotiline .  | 0.179              | 0.367             | 
| chronic kidney failure         | B2M         | to evaluate comparative [beta 2-m]{.gene_color} removal we studied six stable [end-stage renal failure]{.disease_color} patients during high-flux 3-h haemodialysis , haemodia-filtration , and haemofiltration , using acrylonitrile , cellulose triacetate , polyamide and polysulphone capillary devices . | 0.178              | 0.366             | 
| hematologic cancer             | C7          | serum antibody responses to four haemophilus influenzae type b capsular polysaccharide-protein conjugate vaccines ( prp-d , hboc , [c7p]{.gene_color} , and [prp-t )]{.disease_color} were studied and compared in 175 infants , 85 adults and 140 2-year-old children .  | 0.174              | 0.364             | 
| hypertension                   | AVP         | portohepatic pressures , hepatic function , and blood gases in the combination of nitroglycerin and [vasopressin]{.gene_color} : search for additive effects in [cirrhotic portal hypertension]{.disease_color} .  | 0.168              | 0.361             | 
| endogenous depression          | GAD1        | within-individual deflections in gad , physical , and social symptoms predicted later deflections in [depressive symptoms]{.disease_color} , and deflections in depressive symptoms predicted later deflections in [gad]{.gene_color} and separation anxiety symptoms . | 0.149              | 0.349             |  

Table: Contains the bottom ten Disease-associates-Gene confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Gene mentions are highlighted in [blue]{.gene_color}. {#tbl:dg_bottom_ten_table}

| Compound Name          | Disease Name       | Text   | Before Calibration | After Calibration | 
|--------------------|--------------------|---------------------------------------------------------|--------------------|-------------------| 
| Prazosin           | hypertension       | experience with [prazosin]{.compound_color} in the treatment of [hypertension]{.disease_color} .   | 0.997              | 0.961             | 
| Methyldopa         | hypertension       | oxprenolol plus cyclopenthiazide-kcl versus [methyldopa]{.compound_color} in the treatment of [hypertension]{.disease_color} .  | 0.997              | 0.961             | 
| Methyldopa         | hypertension       | atenolol and [methyldopa]{.compound_color} in the treatment of [hypertension]{.disease_color} .  | 0.996              | 0.957             | 
| Prednisone         | asthma             | [prednisone]{.compound_color} and beclomethasone for treatment of [asthma]{.disease_color} .    | 0.995              | 0.953             | 
| Sulfasalazine      | ulcerative colitis | [sulphasalazine]{.compound_color} , used in the treatment of [ulcerative colitis]{.disease_color} , is cleaved in the colon by the metabolic action of colonic bacteria on the diazo bond to release 5-aminosalicylic acid ( 5-asa ) and sulpharidine .    | 0.994              | 0.949             | 
| Prazosin           | hypertension       | letter : [prazosin]{.compound_color} in treatment of [hypertension]{.disease_color} .   | 0.994              | 0.949             | 
| Methylprednisolone | asthma             | use of tao without [methylprednisolone]{.compound_color} in the treatment of severe [asthma]{.disease_color} .    | 0.994              | 0.948             | 
| Budesonide         | asthma             | thus , a regimen of [budesonide]{.compound_color} treatment that consistently attenuates bronchial responsiveness in [asthmatic]{.disease_color} subjects had no effect in these men ; larger and longer trials will be required to establish whether a subgroup of smokers shows a favorable response .   | 0.994              | 0.946             | 
| Methyldopa         | hypertension       | pressor and chronotropic responses to bilateral carotid occlusion ( bco ) and tyramine were also markedly reduced following treatment with [methyldopa]{.compound_color} , which is consistent with the clinical findings that chronic methyldopa treatment in [hypertensive]{.disease_color} patients impairs cardiovascular reflexes . | 0.994              | 0.946             | 
| Fluphenazine       | schizophrenia      | low dose [fluphenazine decanoate]{.compound_color} in maintenance treatment of [schizophrenia]{.disease_color} .  | 0.994              | 0.946             |  

Table: Contains the top ten Compound-treats-Disease confidence scores after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Compound mentions are highlighted in [red]{.compound_color}. {#tbl:cd_top_ten_table}

| Compound Name  | Disease Name              | Text        | Before Calibration | After Calibration | 
|----------------|---------------------------|----------------------------------------------|--------------------|-------------------| 
| Indomethacin   | hypertension              | effects of [indomethacin]{.compound_color} in rabbit [renovascular hypertension]{.disease_color} . | 0.033              | 0.13              | 
| Alprazolam     | panic disorder            | according to logistic regression analysis , the relationships between plasma [alprazolam]{.compound_color} concentration and response , as reflected by number of [panic attacks]{.disease_color} reported , phobia ratings , physicians ' and patients ' ratings of global improvement , and the emergence of side effects , were significant .  | 0.03               | 0.124             | 
| Mestranol      | polycystic ovary syndrome | the binding capacity of plasma testosterone-estradiol-binding globulin ( tebg ) and testosterone ( t ) levels were measured in four women with proved [polycystic ovaries]{.disease_color} and three women with a clinical diagnosis of polycystic ovarian disease before , during , and after administration of norethindrone , 2 mg. , and [mestranol]{.compound_color} , 0.1 mg . | 0.03               | 0.123             | 
| Creatine       | coronary artery disease   | during successful and uncomplicated angioplasty ( ptca ) , we studied the effect of a short lasting [myocardial ischemia]{.disease_color} on plasma creatine kinase , creatine kinase mb-activity , and [creatine]{.compound_color} kinase mm-isoforms ( mm1 , mm2 , mm3 ) in 23 patients .  | 0.028              | 0.12              | 
| Creatine       | coronary artery disease   | in 141 patients with [acute myocardial infarction]{.disease_color} , [creatine]{.compound_color} phosphokinase isoenzyme ( cpk-mb ) was determined by the activation method with dithiothreitol ( rao et al. : clin . | 0.027              | 0.117             | 
| Morphine       | brain cancer              | the tissue to serum ratio of [morphine]{.compound_color} in the [hypothalamus]{.disease_color} , hippocampus , striatum , midbrain and cortex were also smaller in morphine tolerant than in non-tolerant rats .    | 0.026              | 0.115             | 
| Glutathione    | anemia                    | our results suggest that an association between [gsh]{.compound_color} px [deficiency and hemolytic anemia]{.disease_color} need not represent a cause-and-effect relationship .  | 0.026              | 0.114             | 
| Dinoprostone   | stomach cancer            | prostaglandin e2 ( [pge2 )]{.compound_color} - and 6-keto-pgf1 alpha-like immunoactivity was measured in incubates of [forestomach and gastric corpus mucosa]{.disease_color} in ( a ) unoperated rats , ( b ) rats with sham-operation of the kidneys and ( c ) rats with bilateral nephrectomy .   | 0.023              | 0.107             | 
| Creatine       | coronary artery disease   | the value of the electrocardiogram in assessing infarct size was studied using serial estimates of the mb isomer of [creatine]{.compound_color} kinase ( ck mb ) in plasma , serial 35 lead praecordial maps in 28 patients with [anterior myocardial infarction]{.disease_color} , and serial 12 lead electrocardiograms in 17 patients with inferior myocardial infarction .       | 0.022              | 0.105             | 
| Sulfamethazine | multiple sclerosis        | quantitation and confirmation of [sulfamethazine]{.compound_color} residues in swine muscle and liver by lc and [gc/ms]{.disease_color} . | 0.017              | 0.093             |  

Table: Contains the bottom ten Compound-treats-Disease confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Compound mentions are highlighted in [red]{.compound_color}. {#tbl:cd_bottom_ten_table}

| Compound Name                  | Gene Symbol | Text               | Before Calibration | After Calibration | 
|--------------------------------|-------------|-------------------------|--------------------|-------------------| 
| Cyclic Adenosine Monophosphate | B3GNT2      | in sk-n-mc human neuroblastoma cells , the [camp]{.compound_color} response to 10 nm isoproterenol ( iso ) is mediated primarily by [beta 1-adrenergic]{.gene_color} receptors . | 0.903              | 0.93              | 
| Indomethacin                   | AGT         | [indomethacin]{.compound_color} , a potent inhibitor of prostaglandin synthesis , is known to increase the maternal blood pressure response to [angiotensin ii]{.gene_color} infusion . | 0.894              | 0.922             | 
| Tretinoin                      | RXRA        | the vitamin a derivative [retinoic acid]{.compound_color} exerts its effects on transcription through two distinct classes of nuclear receptors , the retinoic acid receptor ( rar ) and the [retinoid x receptor]{.gene_color} ( rxr ) .   | 0.882              | 0.912             | 
| Tretinoin                      | RXRA        | the vitamin a derivative retinoic acid exerts its effects on transcription through two distinct classes of nuclear receptors , the [retinoic acid]{.compound_color} receptor ( rar ) and the [retinoid x receptor]{.gene_color} ( rxr ) .  | 0.872              | 0.903             | 
| D-Tyrosine                     | CSF1        | however , the extent of gap [tyrosine]{.compound_color} phosphorylation induced by [csf-1]{.gene_color} was approximately 10 % of that induced by pdgf-bb in the nih3t3 fibroblasts .   | 0.851              | 0.883             | 
| D-Glutamic Acid                | GLB1        | thus , the negatively charged side chain of [glu-461]{.compound_color} is important for divalent cation binding to [beta-galactosidase]{.gene_color} .  | 0.849              | 0.882             | 
| D-Tyrosine                     | CD4         | second , we use the same system to provide evidence that the physical association of [cd4]{.gene_color} with the tcr is required for effective [tyrosine]{.compound_color} phosphorylation of the tcr zeta-chain subunit , presumably reflecting delivery of p56lck ( lck ) to the tcr . | 0.825              | 0.859             | 
| Calcium Chloride               | TNC         | the possibility that the enhanced length dependence of [ca2]{.compound_color} + sensitivity after cardiac tnc reconstitution was attributable to reduced [tnc]{.gene_color} binding was excluded when the length dependence of partially extracted fast fibres was reduced to one-half the normal value after a 50 % deletion of the native tnc . | 0.821              | 0.855             | 
| Metoprolol                     | KCNMB2      | studies in difi cells of the displacement of specific 125i-cyp binding by nonselective ( propranolol ) , beta 1-selective ( [metoprolol]{.compound_color} and atenolol ) , and beta 2-selective ( ici 118-551 ) antagonists revealed only a single class of [beta 2-adrenergic]{.gene_color} receptors .  | 0.82               | 0.854             | 
| D-Tyrosine                     | PLCG1       | epidermal growth factor ( egf ) or platelet-derived growth factor binding to their receptor on fibroblasts induces tyrosine phosphorylation of plc gamma 1 and stable association of [plc gamma 1]{.gene_color} with the receptor protein [tyrosine]{.compound_color} kinase . | 0.818              | 0.851             |  

Table: Contains the top ten Compound-binds-Gene confidence scores before and after model calbration. Gene mentions are highlighted in [blue]{.gene_color} and Compound mentions are highlighted in [red]{.compound_color}. {#tbl:cg_top_ten_table}

| Compound Name     | Gene Symbol | Text              | Before Calibration | After Calibration | 
|-------------------|-------------|------------------------------------------------------------------|--------------------|-------------------| 
| Deferoxamine      | TF          | the mechanisms of fe uptake have been characterised using 59fe complexes of citrate , nitrilotriacetate , [desferrioxamine]{.compound_color} , and 59fe added to eagle 's minimum essential medium ( mem ) and compared with human transferrin ( [tf )]{.gene_color} labelled with 59fe and iodine-125 .    | 0.02               | 0.011             | 
| Hydrocortisone    | GH1         | group iv patients had normal basal levels of lh and normal lh , [gh]{.gene_color} and [cortisol]{.compound_color} responses .  | 0.02               | 0.011             | 
| Carbachol         | INS         | at the same concentration , however , iapp significantly ( p less than 0.05 ) inhibited [carbachol-stimulated]{.compound_color} ( 10 ( -7 ) m ) release of insulin by 30 % , and cgrp significantly inhibited carbachol-stimulated release of [insulin]{.gene_color} by 33 % when compared with the control group .  | 0.02               | 0.011             | 
| Adenosine         | ME2         | at physiological concentrations , atp , adp , and [amp]{.compound_color} all inhibit the enzyme from atriplex spongiosa and panicum miliaceum ( [nad-me-type]{.gene_color} plants ) , with atp the most inhibitory species .    | 0.019              | 0.01              | 
| Naloxone          | POMC        | specifically , opioids , including 2-n-pentyloxy-2-phenyl-4-methyl-morpholine , [naloxone]{.compound_color} , and [beta-endorphin]{.gene_color} , have been shown to interact with il-2 receptors ( 134 ) and regulate production of il-1 and il-2 ( 48-50 , 135 ) .   | 0.018              | 0.01              | 
| Cortisone acetate | POMC        | sarcoidosis therapy with [cortisone]{.compound_color} and [acth --]{.gene_color} the role of acth therapy .  | 0.017              | 0.009             | 
| Epinephrine       | INS         | thermogenic effect of thyroid hormones : interactions with [epinephrine]{.compound_color} and [insulin]{.gene_color} .  | 0.017              | 0.009             | 
| Aldosterone       | KNG1        | important vasoconstrictor , fluid - and sodium-retaining factors are the [renin-angiotensin-aldosterone]{.compound_color} system , sympathetic nerve activity , and vasopressin ; vasodilator , volume , and sodium-eliminating factors are atrial natriuretic peptide , vasodilator prostaglandins like prostacyclin and prostaglandin e2 , dopamine , [bradykinin]{.gene_color} , and possibly , endothelial derived relaxing factor ( edrf ) . | 0.016              | 0.008             | 
| D-Leucine         | POMC        | cross-reactivities of [leucine-enkephalin]{.compound_color} and [beta-endorphin]{.gene_color} with the eia were less than 0.1 % , while that with gly-gly-phe-met and oxidized gly-gly-phe-met were 2.5 % and 10.2 % , respectively .  | 0.011              | 0.005             | 
| Estriol           | LGALS1      | [ diagnostic value of serial determination of [estriol]{.compound_color} and [hpl]{.gene_color} in plasma and of total estrogens in 24-h-urine compared to single values for diagnosis of fetal danger ] .   | 0.01               | 0.005             | 

Table: Contains the bottom ten Compound-binds-Gene confidence scores before and after model calbration. Gene mentions are highlighted in [blue]{.gene_color} and Compound mentions are highlighted in [red]{.compound_color}. {#tbl:cg_bottom_ten_table}

| Gene1 Symbol | Gene2 Symbol | Text   | Before Calibration | After Calibration | 
|--------------|--------------|----------|--------------------|-------------------| 
| ESR1         | HSP90AA1     | previous studies have suggested that the 90-kda heat shock protein ( [hsp90 )]{.gene_color} interacts with the [er]{.gene_color} , thus stabilizing the receptor in an inactive state .   | 0.812              | 0.864             | 
| TP53         | TP73         | cyclin g interacts with p53 as well as [p73]{.gene_color} , and its binding to [p53]{.gene_color} or p73 presumably mediates downregulation of p53 and p73 .    | 0.785              | 0.837             | 
| TP53         | AKT1         | treatment of c81 cells with ly294002 resulted in an increase in the [p53-responsive]{.gene_color} gene mdm2 , suggesting a role for [akt]{.gene_color} in the tax-mediated regulation of p53 transcriptional activity .  | 0.773              | 0.825             | 
| ABCB1        | NR1I3        | valproic acid induces cyp3a4 and [mdr1]{.gene_color} gene expression by activation of [constitutive androstane receptor]{.gene_color} and pregnane x receptor pathways .  | 0.762              | 0.813             | 
| PTH2R        | PTH2         | thus , the juxtamembrane receptor domain specifies the signaling and binding selectivity of [tip39]{.gene_color} for the [pth2 receptor]{.gene_color} over the pth1 receptor .    | 0.761              | 0.812             | 
| CCND1        | ABL1         | synergy with [v-abl]{.gene_color} depended on a motif in [cyclin d1]{.gene_color} that mediates its binding to the retinoblastoma protein , suggesting that abl oncogenes in part mediate their mitogenic effects via a retinoblastoma protein-dependent pathway .    | 0.757              | 0.808             | 
| CTNND1       | CDH1         | these complexes are formed independently of ddr1 activation and of beta-catenin and [p120-catenin]{.gene_color} binding to [e-cadherin]{.gene_color} ; they are ubiquitous in epithelial cells . | 0.748              | 0.798             | 
| CSF1         | CSF1R        | this is in agreement with current thought that the [c-fms]{.gene_color} proto-oncogene product functions as the [csf-1]{.gene_color} receptor specific to this pathway . | 0.745              | 0.795             | 
| EZR          | CFTR         | without [ezrin]{.gene_color} binding , the cytoplasmic tail of [cftr]{.gene_color} only interacts strongly with the first amino-terminal pdz domain to form a 1:1 c-cftr .  | 0.732              | 0.78              | 
| SRC          | PIK3CG       | we have demonstrated that the sh2 ( [src]{.gene_color} homology 2 ) domains of the 85 kda subunit of pi-3k are sufficient to mediate binding of the [pi-3k]{.gene_color} complex to tyrosine phosphorylated , but not non-phosphorylated il-2r beta , suggesting that tyrosine phosphorylation is an integral component of the activation of pi-3k by the il-2r . | 0.731              | 0.78              | 

Table: Contains the top ten Gene-interacts-Gene confidence scores before and after model calbration. Both gene mentions highlighted in [blue]{.gene_color}. {#tbl:gg_top_ten_table}

| Gene1 Symbol | Gene2 Symbol | Text  | Before Calibration | After Calibration | 
|--------------|--------------|----------|--------------------|-------------------| 
| AGTR1        | ACE          | result ( s ) : the luteal tissue is the major site of ang ii , [ace]{.gene_color} , [at1r]{.gene_color} , and vegf , with highest staining intensity found during the midluteal phase and at pregnancy . | 0.009              | 0.003             | 
| ABCE1        | ABCF2        | in relation to normal melanocytes , abcb3 , abcb6 , abcc2 , abcc4 , [abce1]{.gene_color} and [abcf2]{.gene_color} were significantly increased in melanoma cell lines , whereas abca7 , abca12 , abcb2 , abcb4 , abcb5 and abcd1 showed lower expression levels . | 0.008              | 0.002             | 
| IL4          | IFNG         | in contrast , il-13ralpha2 mrna expression was up-regulated by [ifn-gamma]{.gene_color} plus [il-4]{.gene_color} . | 0.007              | 0.002             | 
| FCAR         | CD79A        | we report here the presence of circulating soluble fcalphar ( [cd89 )]{.gene_color} - [iga]{.gene_color} complexes in patients with igan .    | 0.007              | 0.002             | 
| IL4          | VCAM1        | similarly , [il-4]{.gene_color} induced [vcam-1]{.gene_color} expression and augmented tnf-alpha-induced expression on huvec but did not affect vcam-1 expression on hdmec .  | 0.007              | 0.002             | 
| IL2          | IFNG         | prostaglandin e2 at priming of naive cd4 + t cells inhibits acquisition of ability to produce [ifn-gamma]{.gene_color} and [il-2]{.gene_color} , but not il-4 and il-5 . | 0.006              | 0.002             | 
| IL2          | FOXP3        | il-1b promotes tgf-b1 and [il-2]{.gene_color} dependent [foxp3]{.gene_color} expression in regulatory t cells . | 0.006              | 0.002             | 
| IL2          | IFNG         | the detailed distribution of lymphokine-producing cells showed that [il-2]{.gene_color} and [ifn-gamma-producing]{.gene_color} cells were located mainly in the follicular areas .    | 0.005              | 0.001             | 
| IFNG         | IL10         | results : we found weak mrna expression of interleukin-4 ( il-4 ) and il-5 , and strong expression of il-6 , [il-10]{.gene_color} and [ifn-gamma]{.gene_color} before therapy . | 0.005              | 0.001             | 
| PIK3R1       | PTEN         | both [pten]{.gene_color} ( [pi3k]{.gene_color} antagonist ) and pp2 ( unspecific phosphatase ) were down-regulated .  | 0.005              | 0.001             |  

Table: Contains the bottom ten Gene-interacts-Gene confidence scores before and after model calbration. Both gene mentions highlighted in [blue]{.gene_color}. {#tbl:gg_bottom_ten_table}

### Baseline Comparison

![
Comparion between our model and CoCoScore model [@IGXdryzB].
We report both model's performance in terms of AUROC and AUPR.
Our model achieves comparable performance against CoCoScore in terms of AUROC.
As for AUPR, CoCoScore consistently outperforms our model except for CtD. 
](https://raw.githubusercontent.com/danich1/snorkeling/0149086785b19f9429c92565d650e9d049c136ff/figures/literature_models/model_comparison.png){#fig:cocoscore_comparison}

Once our discriminator model is calibrated, we grouped sentences based on mention pair (edges).
We assigned each edge the maximum score over all grouped sentences and compared our model's ability to predict pairs in our test set to a previously published baseline model [@IGXdryzB].
Performance is reported in terms of AUROC and AUPR (Figure {@fig:cocoscore_comparison}).
Across edge types our model shows comparable performance against the baseline in terms of AUROC.
Regarding AUPR, our model shows hindered performance against the baseline.
The exception for both cases is CtD where our model performs better than the baseline.

### Reconstructing Hetionet

![
A scatter plot showing the number of edges (log scale) we can add or recall at specified precision levels. 
The blue depicts edges existing in hetionet and the orange depicts how many novel edges can be added.
](https://raw.githubusercontent.com/danich1/snorkeling/0149086785b19f9429c92565d650e9d049c136ff/figures/edge_prediction_experiment/edges_added.png){#fig:hetionet_reconstruction}

We evaluated how many edges we can recall/add to Hetionet v1 (Supplemental Figure {@fig:hetionet_reconstruction} and Table {@tbl:edge_prediction_tbl}).
In our evaluation we used edges assigned to our test set.
Overall, we can recall a small amount of edges at high precision thresholds.
A key example is CbG and GiG where we recalled only one exisiting edge at 100% precision.
Despite the low recall, we are still able to add novel edges to DaG and CtD while retaining modest precision.

| Edge Type | Source Node             | Target Node               | Gen Model Prediction | Disc Model Prediction | Number of Sentences | Text   | 
|--------------|----------------------|------------------------|----------------------|-----------------------|---------------------|-----------------------------------------| 
| [D]{.disease_color}a[G]{.gene_color}       | lung cancer             | VEGFA                     | 1.000                | 0.912                 | 3293                | conclusion : the plasma [vegf]{.gene_color} level is increased in [nsclc]{.disease_color} patients with approximate1y one fourth to have cancer cells in the peripheral blood.                                                                                                                                                                                  | 
| [D]{.disease_color}a[G]{.gene_color}       | hematologic cancer      | TP53                      | 1.000                | 0.905                 | 8660                | mutations of the [p53]{.gene_color} gene were found in four cases of [cml]{.disease_color} in blastic crisis ( bc ).                                                                                                                                                                                                                                            | 
| [D]{.disease_color}a[G]{.gene_color}       | obesity                 | MC4R                      | 1.000                | 0.901                 | 1493                | several mutations in the [melanocortin 4 receptor]{.gene_color} gene are associated with [obesity]{.disease_color} in chinese children and adolescents.                                                                                                                                                                                                         | 
| [D]{.disease_color}a[G]{.gene_color}       | Alzheimer's disease     | VLDLR                     | 1.000                | 0.886                 | 86                  | the 5-repeat allele in the [very-low-density lipoprotein receptor]{.gene_color} gene polymorphism is not increased in sporadic [alzheimer 's disease]{.disease_color} in japanese.                                                                                                                                                                              | 
| [D]{.disease_color}a[G]{.gene_color}       | lung cancer             | XRCC1                     | 1.000                | 0.885                 | 662                 | results : [xrcc1]{.gene_color} gene polymorphism is associated with increased risk of [lung cancer]{.disease_color} when the arg/arg genotype was used as the reference group.                                                                                                                                                                                  | 
| [D]{.disease_color}a[G]{.gene_color}       | prostate cancer         | ESR1                      | 1.000                | 0.883                 | 500                 | conclusion : these results suggest that variants of the ggga polymorphism from the [estrogen receptor alpha]{.gene_color} gene may be associated with an increased risk of developing [prostate cancer]{.disease_color}.                                                                                                                                        | 
| [D]{.disease_color}a[G]{.gene_color}       | breast cancer           | REG1A                     | 1.000                | 0.878                 | 37                  | conclusion : high levels of [reg1a]{.gene_color} expression within tumors are an independent predictor of poor prognosis in patients with [breast cancer]{.disease_color}.                                                                                                                                                                                      | 
| [D]{.disease_color}a[G]{.gene_color}       | breast cancer           | INSR                      | 1.000                | 0.877                 | 200                 | we have previously reported that [insulin receptor]{.gene_color} expression is increased in human [breast cancer]{.disease_color} specimens ( v. papa et al. , j. clin.                                                                                                                                                                                         | 
| [D]{.disease_color}a[G]{.gene_color}       | rheumatoid arthritis    | AR                        | 1.000                | 0.877                 | 53                  | conclusion : our results suggest no correlation between cag repeat polymorphism in the [ar]{.gene_color} gene and response to treatment with lef in women with [ra]{.disease_color}.                                                                                                                                                                            | 
| [D]{.disease_color}a[G]{.gene_color}       | coronary artery disease | CTLA4                     | 1.000                | 0.875                 | 12                  | conclusion : the g/g genotype polymorphism of the [ctla-4]{.gene_color} gene is associated with increased risk of [ami]{.disease_color}.                                                                                                                                                                                                                        | 
| [C]{.compound_color}t[D]{.disease_color}       | Zonisamide              | epilepsy syndrome         | 1.000                | 0.943                 | 1011                | adjunctive [zonisamide]{.compound_color} therapy in the long-term treatment of children with [partial epilepsy]{.disease_color} : results of an open-label extension study of a phase iii , randomized , double-blind , placebo-controlled trial.                                                                                                               | 
| [C]{.compound_color}t[D]{.disease_color}       | Metformin               | polycystic ovary syndrome | 1.000                | 0.942                 | 3217                | in the present study , 23 [pcos]{.disease_color} subjects [ mean ( + / - se ) body mass index 30.0 + / -1.1 kg/m2 ] were randomly assigned to double-blind treatment with [metformin]{.compound_color} ( 500 mg tid ) or placebo for 6 months , while maintaining their usual eating habits.                                                                    | 
| [C]{.compound_color}t[D]{.disease_color}       | Piroxicam               | rheumatoid arthritis      | 1.000                | 0.928                 | 184                 | methods : a double-blind , randomized , crossover trial in 49 patients with active [ra]{.disease_color} compared 6 weeks of treatment with tenidap ( 120 mg/day ) versus 6 weeks of treatment with [piroxicam]{.compound_color} ( 20 mg/day ).                                                                                                                  | 
| [C]{.compound_color}t[D]{.disease_color}       | Irinotecan              | stomach cancer            | 1.000                | 0.918                 | 968                 | randomized phase ii trial of first-line treatment with tailored [irinotecan]{.compound_color} and s-1 therapy versus s-1 monotherapy for advanced or recurrent [gastric carcinoma]{.disease_color} ( jfmc31-0301 ).                                                                                                                                             | 
| [C]{.compound_color}t[D]{.disease_color}       | Treprostinil            | hypertension              | 1.000                | 0.913                 | 536                 | oral [treprostinil]{.compound_color} for the treatment of [pulmonary arterial hypertension]{.disease_color} in patients receiving background endothelin receptor antagonist and phosphodiesterase type 5 inhibitor therapy ( the freedom-c2 study ) : a randomized controlled trial.                                                                            | 
| [C]{.compound_color}t[D]{.disease_color}       | Colchicine              | gout                      | 1.000                | 0.911                 | 78                  | this is the first in vivo data to provide a biological rationale that supports the implementation of low dose , non-toxic , [colchicine]{.compound_color} therapy for the treatment of [gouty arthritis]{.disease_color}.                                                                                                                                       | 
| [C]{.compound_color}t[D]{.disease_color}       | Propranolol             | stomach cancer            | 1.000                | 0.898                 | 45                  | 74 cirrhotic patients with a history of variceal or [gastric bleeding]{.disease_color} were randomly assigned to treatment with [propranolol]{.compound_color} ( 40 to 360 mg/day ) or placebo.                                                                                                                                                                 | 
| [C]{.compound_color}t[D]{.disease_color}       | Reboxetine              | endogenous depression     | 1.000                | 0.894                 | 439                 | data were obtained from four short-term ( 4-8-week ) , randomized , placebo-controlled trials of [reboxetine]{.compound_color} for the treatment of [mdd]{.disease_color}.                                                                                                                                                                                      | 
| [C]{.compound_color}t[D]{.disease_color}       | Diclofenac              | ankylosing spondylitis    | 1.000                | 0.892                 | 61                  | comparison of two different dosages of celecoxib with [diclofenac]{.compound_color} for the treatment of active [ankylosing spondylitis]{.disease_color} : results of a 12-week randomised , double-blind , controlled study.                                                                                                                                   | 
| [C]{.compound_color}t[D]{.disease_color}       | Tapentadol              | osteoarthritis            | 1.000                | 0.880                 | 29                  | driving ability in patients with severe chronic low back or [osteoarthritis]{.disease_color} knee pain on stable treatment with [tapentadol]{.compound_color} prolonged release : a multicenter , open-label , phase 3b trial.                                                                                                                                  | 
| [C]{.compound_color}b[G]{.gene_color}       | Dexamethasone           | NR3C1                     | 1.000                | 0.850                 | 1119                | submicromolar free calcium modulates [dexamethasone]{.compound_color} binding to the [glucocorticoid receptor]{.gene_color}.                                                                                                                                                                                                                                    | 
| [C]{.compound_color}b[G]{.gene_color}       | Vitamin A               | RBP4                      | 1.000                | 0.807                 | 5512                | the authors give serum [retinol]{.compound_color} binding protein ( [rbp )]{.gene_color} normal values , established by immunonephelometry , for two healthy populations in their hospital laboratory.                                                                                                                                                          | 
| [C]{.compound_color}b[G]{.gene_color}       | D-Proline               | IGFBP4                    | 1.000                | 0.790                 | 1                   | the insulin-like growth factor-i-stimulated [l-proline]{.compound_color} uptake was inhibited by one of its binding protein , [insulin-like growth factor binding protein-4]{.gene_color} , in a concentration-dependent manner.                                                                                                                                | 
| [C]{.compound_color}b[G]{.gene_color}       | Sucrose                 | AR                        | 0.996                | 0.789                 | 37                  | the amount ( maximal binding capacity of 24 to 30 fmol/mg protein ) and hormone binding affinity ( half-maximal saturation of 0.2 nm ) of the [androgen receptor]{.gene_color} in cultured skin fibroblasts was normal , but the receptor was qualitatively abnormal as evidenced by instability on [sucrose]{.compound_color} density gradient centrifugation. | 
| [C]{.compound_color}b[G]{.gene_color}       | D-Lysine                | PLG                       | 1.000                | 0.787                 | 403                 | in both elisa and rocket immunoelectrophoresis systems , complex formation was inhibited by 10 mm epsilon-amino-n-caproic acid , implying that there is a role for the [lysine]{.compound_color} binding sites of [plg]{.gene_color} in mediating the interaction.                                                                                              | 
| [C]{.compound_color}b[G]{.gene_color}       | Adenosine               | INSR                      | 1.000                | 0.785                 | 129                 | these findings demonstrate basal state binding of [atp]{.compound_color} to the ckd leading to cis-autophosphorylation and novel basal state regulatory interactions among the subdomains of the [insulin receptor]{.gene_color} kinase.                                                                                                                        | 
| [C]{.compound_color}b[G]{.gene_color}       | Adenosine               | PLK1                      | 1.000                | 0.783                 | 104                 | most kinase inhibitors interact with the [atp]{.compound_color} binding site on [plk1]{.gene_color} , which is highly conserved.                                                                                                                                                                                                                                | 
| [C]{.compound_color}b[G]{.gene_color}       | Calcium Chloride        | ITPR3                     | 0.995                | 0.777                 | 1954                | control of [ca2]{.compound_color} + influx in human neutrophils by inositol 1,4,5-trisphosphate ( ip3 ) binding : differential effects of micro-injected [ip3 receptor]{.gene_color} antagonists.                                                                                                                                                               | 
| [C]{.compound_color}b[G]{.gene_color}       | D-Arginine              | C5AR1                     | 1.000                | 0.775                 | 808                 | thus , selected out of a multiplicity of possibilities by the natural binding partner , [arg37]{.compound_color} as well as arg40 appear to be anchor residues in binding to the [c5a receptor]{.gene_color}.                                                                                                                                                   | 
| [C]{.compound_color}b[G]{.gene_color}       | Ticagrelor              | P2RY12                    | 1.000                | 0.773                 | 322                 | purpose : [ticagrelor]{.compound_color} is a reversibly binding [p2y12]{.gene_color} receptor antagonist used clinically for the prevention of atherothrombotic events in patients with acute coronary syndromes ( acs ).                                                                                                                                       | 
| [G]{.gene_color}i[G]{.gene_color}       | ABL1                    | ABL1                      | 0.999                | 0.600                 | 9572                | the acquired resistance in patients who failed to respond to imatinib seemed to be induced by several point mutations in the [bcr-abl]{.gene_color} gene , which were likely to affect the binding of imatinib with [bcr-abl]{.gene_color}.                                                                                                                     | 
| [G]{.gene_color}i[G]{.gene_color}       | TP63                    | TP53                      | 1.000                | 0.595                 | 2557                | [tp63]{.gene_color} , a member of the [p53]{.gene_color} gene family gene , encodes the np63 protein and is one of the most frequently amplified genes in squamous cell carcinomas ( scc ) of the head and neck ( hnscc ) and lungs ( lusc ).                                                                                                                   | 
| [G]{.gene_color}i[G]{.gene_color}       | FERMT1                  | FERMT1                    | 0.004                | 0.590                 | 194                 | ks is caused by mutations in the [fermt1]{.gene_color} gene encoding [kindlin-1]{.gene_color}.                                                                                                                                                                                                                                                                  | 
| [G]{.gene_color}i[G]{.gene_color}       | GRN                     | GRN                       | 1.000                | 0.590                 | 3842                | background : mutations in the [progranulin]{.gene_color} gene ( [pgrn )]{.gene_color} have recently been identified as a cause of frontotemporal lobar degeneration with ubiquitin-positive inclusions ( ftld-u ) in some families.                                                                                                                             | 
| [G]{.gene_color}i[G]{.gene_color}       | FASN                    | EP300                     | 0.999                | 0.589                 | 6                   | here , we demonstrated that [p300]{.gene_color} binds to and increases histone h3 lysine 27 acetylation ( h3k27ac ) in the [fasn]{.gene_color} gene promoter.                                                                                                                                                                                                   | 
| [G]{.gene_color}i[G]{.gene_color}       | SETBP1                  | SETBP1                    | 1.000                | 0.588                 | 354                 | the critical deleted region contains [setbp1]{.gene_color} gene ( [set binding protein 1 )]{.gene_color}.                                                                                                                                                                                                                                                       | 
| [G]{.gene_color}i[G]{.gene_color}       | BCL2                    | BAK1                      | 0.118                | 0.587                 | 1220                | different expression patterns of [bcl-2]{.gene_color} family genes in breast cancer by estrogen receptor status with special reference to pro-apoptotic [bak]{.gene_color} gene.                                                                                                                                                                                | 
| [G]{.gene_color}i[G]{.gene_color}       | SP1                     | INSR                      | 0.948                | 0.587                 | 23                  | thus , the efficient expression of the human [insulin receptor]{.gene_color} gene possibly requires the binding of transcriptional factor [sp1]{.gene_color} to four g-c boxes located -593 to -618 base pairs upstream of the atg translation initiation codon.                                                                                                | 
| [G]{.gene_color}i[G]{.gene_color}       | ABCD1                   | ABCD1                     | 1.000                | 0.586                 | 410                 | x-linked adrenoleukodystrophy ( x-ald ) is caused by mutations in the [abcd1]{.gene_color} gene encoding the peroxisomal abc transporter adrenoleukodystrophy protein ( [aldp )]{.gene_color}.                                                                                                                                                                  | 
| [G]{.gene_color}i[G]{.gene_color}       | CYP1A1                  | AHR                       | 0.996                | 0.586                 | 1940                | the liganded [ah receptor]{.gene_color} activates transcription by binding to a specific dna-recognition motif within a dioxin-responsive enhancer upstream of the [cyp1a1]{.gene_color} gene.                                                                                                                                                                  |
Table: Contains the top ten predictions for each edge type. Highlighted words represent entities mentioned within the given sentence. {#tbl:edge_prediction_tbl}

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

  <link rel="alternate" type="text/html" href="https://greenelab.github.io/text_mined_hetnet_manuscript/v/e0278f840c7e4cccf89df47b193c15bb27c43ceb/" />

  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/text_mined_hetnet_manuscript/v/e0278f840c7e4cccf89df47b193c15bb27c43ceb/" />

  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/text_mined_hetnet_manuscript/v/e0278f840c7e4cccf89df47b193c15bb27c43ceb/manuscript.pdf" />

  <meta property="og:type" content="article" />

  <meta property="twitter:card" content="summary_large_image" />

  <meta property="og:image" content="https://github.com/greenelab/text_mined_hetnet_manuscript/raw/e0278f840c7e4cccf89df47b193c15bb27c43ceb/thumbnail.png" />

  <meta property="twitter:image" content="https://github.com/greenelab/text_mined_hetnet_manuscript/raw/e0278f840c7e4cccf89df47b193c15bb27c43ceb/thumbnail.png" />

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
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/e0278f840c7e4cccf89df47b193c15bb27c43ceb/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@e0278f8](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/e0278f840c7e4cccf89df47b193c15bb27c43ceb)
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

Knowledge bases are important resources that hold complex structured and unstructured information. 
These resources have been used in important tasks such as network analysis for drug repurposing discovery [@u8pIAt5j; @bPvC638e; @O21tn8vf] or as a source of training labels for text mining systems [@EHeTvZht; @CVHSURuI; @HS4ARwmZ]. 
Populating knowledge bases often requires highly-trained scientists to read biomedical literature and summarize the results [@N1Ai0gaI].
This time consuming process is referred to as manual curation.
In 2007 researchers estimated that filling a knowledge base via manual curation would require approximately 8.4 years to complete [@UdzvLgBM]. 
The rate of publications continues to exponentially increase [@1DBISRlwN], so using only manual curation to fully populate a knowledge base has become impractical.  

Relationship extraction has been studied as a solution towards handling the challenge posed by an exponentially growing body of literature [@N1Ai0gaI].
This process consists of creating an expert system to automatically scan, detect and extract relationships from textual sources.
Typically, these systems utilize machine learning techniques that require large corpora of well-labeled training data.
These corpora are difficult to obtain, because they are constructed via particularly detailed manual curation.
Distant supervision is a technique designed to sidestep the dependence on manual curation and quickly generate large training datasets.
This technique makes the assumption that positive examples established in selected databases can be applied to any sentence that contains them [@EHeTvZht].
The central problem with this technique is that generated labels are often of low quality which results in an immense amount of false positives [@mwM58zzr].  

Ratner et al. [@5Il3kN32] recently introduced "data programming" as a solution.
Data programming is a paradigm that combines distant supervision with simple rules and heuristics written as small programs called label functions.
These label functions are consolidated via a noise aware generative model that is designed to produce training labels for large datasets.
Using this paradigm can dramatically reduce the time required to obtain sufficient training data; however, writing a useful label function requires a significant amount of time and error analysis.
This dependency makes constructing a knowledge base with a myriad of heterogenous relationships nearly impossible as tens or possibly hundreds of label functions are required per relationship type.  

In this paper, we seek to accelerate the label function creation process by measuring the extent to which label functions can be re-used across different relationship types.
We hypothesize that sentences describing one relationship type may share linguistic features such as keywords or sentence structure with sentences describing other relationship types.
We conduct a series of experiments to determine the degree to which label function re-use enhanced performance over distant supervision alone.
We focus on relationships that indicate similar types of physical interactions (i.e., gene-binds-gene and compound-binds-gene) as well as different types (i.e., disease-associates-gene and compound-treats-disease).
Re-using label functions could dramatically reduce time required to populate a knowledge base with a multitude of heterogeneous relationships.

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

#### Label Function Categories

Label functions can be constructed in a multitude of ways; however,  many label functions share similar characteristics with one another.  
We group these characteristics into the following categories: databases, text patterns and domain heuristics.
Most of our label functions fall into the text pattern category, while the others were distributed across the database and domain heuristic categories (Table {@tbl:label-functions}).
We describe each category and provide an example using the candidate sentence: "[PTK6]{.gene_color} may be a novel therapeutic target for [pancreatic cancer]{.disease_color}.".

**Databases**: These label functions incorporate existing databases to generate a signal, as seen in distant supervision [@EHeTvZht].
These functions detect if a candidate sentence's co-mention pair is present in a given database.
If the pair is present, our label function emits a positive label and abstains otherwise.
If the pair is not present in any existing database, a separate label function emits a negative label.
We used a separate label function to prevent a label imbalance problem that we encountered during development: emitting positives and negatives from the same label function causes downstream classifiers to generate almost exclusively negative predictions.

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

**Domain Heuristics**: These label functions used results from published text-based analyses to generate a signal. 
We used dependency path cluster themes generated by Percha et al. [@CSiMoOrI].
If a candidate sentence's dependency path belonged to a previously generated cluster, then the label function emitted a positive label and abstained otherwise.

$$
\Lambda_{DH}(\color{#875442}{D}, \color{#02b3e4}{G}) = \begin{cases}
    1 & Candidate \> Sentence \in Cluster \> Theme\\
    0 & otherwise \\
    \end{cases}
$$

**Text Patterns**: These label functions are designed to use keywords and sentence context to generate a signal. 
For example, a label function could focus on the number of words between two mentions or focus on the grammatical structure of a sentence.
These functions emit a positive or negative label depending on the context.

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

Each text pattern label function was constructed by manual examination of sentences within the training set.
For example, in the candidate sentence above one would extract the keywords "novel therapeutic target" and incorporate them in a text pattern label function.
After initial construction, we tested and augmented the label function using sentences in the tune set.
We repeated the above process for each label function in our repertoire. 

| Relationship | Databases (DB) | Text Patterns (TP) | Domain Heuristics (DH) |
| --- | :---: | :---: | :---: |
| DaG | 7 | 20 | 10 | 
| CtD | 3 | 15 | 7 |
| CbG | 9 | 13 | 7 | 
| GiG | 9 | 20 | 8 | 

Table: The distribution of each label function per relationship. {#tbl:label-functions} 

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

Creating label functions is a labor intensive process that can take days to accomplish.
We sought to accelerate this process by measuring the extent to which label functions can be reused.
Our hypothesis was that certain edge types share similar linguistic features such as keywords and/or sentence structure.
This shared characteristic would make certain edge types amenable to label function reuse.
We designed a set of experiments to test this hypothesis on an individual level (edge vs edge) as well as a global level (collective pool of sources). 
We report results in terms of AUROC (Figures {@fig:auroc_gen_model_test_set} and {@fig:auroc_grabbag_gen_model_test_set}) and AUPR (Supplemental Figure {@fig:aupr_gen_model_test_set} and {@fig:aupr_grabbag_gen_model_test_set}).

Performance increases when edge-specific label functions are added to an edge-specific baseline model, while label function reusability shows modest results.
The quintessential example of the overarching trend is the Compound treats Disease (CtD) edge type, where edge-specific label functions always outperformed transferred label functions.
However, there are hints of label function transferability for selected edge types and label function sources. 
Performance increases as more CbG label functions are incorporated to the GiG baseline model and vise-versa.
This suggests that sentences for GiG and CbG may share similar linguistic features or terminology that allows for label functions to be reused.
Edge-specific Disease associates Gene (DaG) label functions did not improve performance over label functions drawn from other edge types.
Overall, only CbG and GiG show significant signs of reusability which suggests label functions could be shared between the two edge types.

![
Edge-specific label functions are better performing than edge-mismatch label functions but certain mismatch situations show signs of successful transfer.
Each line plot header depicts the edge type the generative model is trying to predict, while the colors represent the source of label functions.
For example orange represents sampling label functions designed to predict the Compound treats Disease (CtD) edge type.
The x axis shows the number of randomly sampled label functions being incorporated onto the database only baseline model (point at 0).
The y axis shows area under the receiver operating curve (AUROC).
Each point on the plot shows the average of 50 sample runs, while the error bars show the 95% confidence intervals of all runs.
The baseline and “All” data points consist of sampling from the entire fixed set of label functions.
](https://raw.githubusercontent.com/danich1/snorkeling/86037d185a299a1f6dd4dd68605073849c72af6f/figures/label_sampling_experiment/transfer_test_set_auroc.png){#fig:auroc_gen_model_test_set}

We found that sampling from all label function sources at once usually underperformed relative to edge-specific label functions (Figure {#fig:auroc_grabbag_gen_model_test_set}).
As more label functions were sampled, the gap between edge-specific sources and all sources widened.
CbG is a prime example of this trend (Figure {#fig:auroc_grabbag_gen_model_test_set}), while CtD and GiG show a similar but milder trend.
DaG was the exception to the general rule: the pooled set of label functions improved performance over the edge-specific ones, which aligns with the previously observed results for individual edge types (Figure {#fig:auroc_gen_model_test_set}).
The decreasing trend when pooling all label functions supports the notion that label functions cannot easily transfer between edge types (exception being CbG on GiG and vise versa).

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

### Discriminative Model Performance

![
The discriminator model usually improves at a faster rate than the generative model as more edge-specific label function are included.
The line plot headers represents the specific edge type the discriminator model is trying to predict.
The x-axis shows the number of randomly sampled label functions that are incorporated on top of the baseline model (point at 0).
The y axis shows the area under the receiver operating curve (AUROC).
Each datapoint represents the average of each 50 sample run and the error bars represent the 95% confidence interval of each run.
The baseline and “All” data points consist of sampling from the entire fixed set of label functions.
This makes the error bars appear flat.
](https://raw.githubusercontent.com/danich1/snorkeling/1941485a02c8aa9972c67d8f9d3ff96acb0f3b7b/figures/disc_model_experiment/disc_model_test_auroc.png){#fig:auroc_discriminative_model_performance}

The discriminator model is designed to augment performance over the generative model by incorporating textual features along with estimated training labels.
The discriminative model is a piecewise convolutional neural network trained over word embeddings (See Methods).
We found that the discriminative model generally out-performed the generative model as more edge-specific label functions are incorporated (Figure {@fig:auroc_discriminative_model_performance} and Supplemental Figure {@fig:aupr_discriminative_model_performance}).
The discriminator model's performance is often poorest when very few edge-specific label functions are added to the baseline model (seen in Disease associates Gene (DaG), Compound binds Gene (CbG) and Gene interacts Gene (GiG)). 
This suggests that generative models trained with more label functions produce outputs that are more suitable for training discriminative models.
An exception to this trend is Compound treats Disease (CtD) where the discriminator model out-performs the generative model at all levels of sampling.
We observed the opposite trend with the Compound-binds-Gene (CbG) edges: the discriminator model was always poorer or indistinguishable from the generative model.
Interestingly, the AUPR for CbG plateaus below the generative model and the decreases when all edge-specific label functions are used (Supplemental Figure {@fig:aupr_discriminative_model_performance}).
This suggests that the discriminator model might be predicting more false positives in this setting.
Incorporating more edge-specific label functions usually improves performance for the discriminator model over the generator model.

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
As the rate of publications increases, relying on manual curation alone becomes impractical.
Data programming, a paradigm that uses label functions as a means to speed up the annotation process, can be used as a solution for this problem.
An obstacle for this paradigm is creating useful label functions, which takes a considerable amount of time. 
We tested the feasibility of reusing label functions as a way to reduce the total number of label functions required for strong prediction performance.
We conclude that label functions may be re-used with closely related edge types, but that re-use does not improve performance for most pairings.
The discriminative model's performance improves as more edge-specific label functions are incorporated into the generative model; however, we did notice that performance greatly depends on the generative model.

This work sets up the foundation for creating a common framework that mines text to create edges.
Within this framework we would continuously ingest new knowledge as novel findings are published, while providing a single confidence score for an edge via sentence score consolidation.
As opposed to many existing knowledge graphs, for example Hetionet where text-derived edges generally cannot be exactly attributed to excerpts from literature [@O21tn8vf; @L2B5V7XC], our approach has the potential to annotate each edge based on its source sentences.
In addition, edges generated with this approach would be unencumbered from upstream licensing or copyright restrictions, enabling openly licensed hetnets at a scale not previously possible [@4G0GW8oe; @137tbemL9; @1GwdMLPbV].
New multitask learning [@9Jo1af7Z] strategies may make it even more practical to reuse label functions to construct continuously updating literature-derived knowledge graphs.


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

### Generative Model Using Randomly Sampled Label Functions

#### Individual Sources

![
Edge-specific label functions improves performance over edge-mismatch label functions.
Each line plot header depicts the edge type the generative model is trying to predict, while the colors represent the source of label functions.
For example orange represents sampling label functions designed to predict the Compound treats Disease (CtD) edge type.
The x axis shows the number of randomly sampled label functions being incorporated onto the database only baseline model (point at 0).
The y axis shows area under the precision recall curve (AUPR).
Each point on the plot shows the average of 50 sample runs, while the error bars show the 95% confidence intervals of all runs.
The baseline and “All” data points consist of sampling from the entire fixed set of label functions.
](https://raw.githubusercontent.com/danich1/snorkeling/86037d185a299a1f6dd4dd68605073849c72af6f/figures/label_sampling_experiment/transfer_test_set_aupr.png){#fig:aupr_gen_model_test_set}

#### Collective Pool of Sources 

![
Using all label functions generally hinders generative model performance.
Each line plot header depicts the edge type the generative model is trying to predict, while the colors represent the source of label functions.
For example, orange represents sampling label functions designed to predict the Compound treats Disease (CtD) edge type.
The x axis shows the number of randomly sampled label functions being incorporated onto the database only baseline model (point at 0).
The y axis shows area under the precision recall curve (AUPR).
Each point on the plot shows the average of 50 sample runs, while the error bars show the 95% confidence intervals of all runs.
The baseline and “All” data points consist of sampling from the entire fixed set of label functions.
](https://raw.githubusercontent.com/danich1/snorkeling/86037d185a299a1f6dd4dd68605073849c72af6f/figures/label_sampling_experiment/all_lf_test_set_aupr.png){#fig:aupr_grabbag_gen_model_test_set}

### Discriminative Model Performance

![
The discriminator model improves performance as the number of edge-specific label functions is added to the baseline model.
The line plot headers represents the specific edge type the discriminator model is trying to predict.
The x-axis shows the number of randomly sampled label functions incorporated on top of the baseline model (point at 0).
The y axis shows the area under the precision recall curve (AUPR).
Each datapoint shows the average of each sample runs, while the error bars represents the 95% confidence interval at each point.
The baseline and “All” data points consist of sampling from the entire fixed set of label functions.
This makes the error bars appear flat.
](https://raw.githubusercontent.com/danich1/snorkeling/1941485a02c8aa9972c67d8f9d3ff96acb0f3b7b/figures/disc_model_experiment/disc_model_test_aupr.png){#fig:aupr_discriminative_model_performance}

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

#### Top Ten Sentences for Each Edge Type

| Edge Type | Source Node | Target Node | Generative Model Prediction | Discriminative Model Prediction | In Hetionet? |  Number of Sentences | Text     | 
|-----------|-------------|-------------|-----------------------------|---------------------------------|--------------|----------------------|----------|
| [D]{.disease_color}a[G]{.gene_color}       | urinary bladder cancer | TP53        | 1                           | 0.945                           | 2112         | Existing             | conclusion : our findings indicate that the dsp53-285 can upregulate wild-type [p53]{.gene_color} expression in human [bladder cancer]{.disease_color} cells through rna activation , and suppresses cells proliferation and metastasis in vitro and in vivo . | 
| [D]{.disease_color}a[G]{.gene_color}       | ovarian cancer         | EGFR        | 1                           | 0.937                           | 1330         | Existing             | conclusion : our data showed that increased expression of [egfr]{.gene_color} is associated with poor prognosis of patients with [eoc]{.disease_color} and dacomitinib may act as a novel , useful chemotherapy drug .  | 
| [D]{.disease_color}a[G]{.gene_color}       | stomach cancer         | TP53        | 1                           | 0.937                           | 2679         | Existing             | conclusion : this meta-analysis suggests that [p53]{.gene_color} arg72pro polymorphism is associated with increased risk of [gastric cancer]{.disease_color} in asians .  | 
| [D]{.disease_color}a[G]{.gene_color}       | lung cancer            | TP53        | 1                           | 0.936                           | 6813         | Existing             | conclusion : these results suggest that high expression of the [p53]{.gene_color} oncoprotein is a favorable prognostic factor in a subset of patients with [nsclc]{.disease_color} .  | 
| [D]{.disease_color}a[G]{.gene_color}       | breast cancer          | TCF7L2      | 1                           | 0.936                           | 56           | Existing             | this meta-analysis demonstrated that [tcf7l2]{.gene_color} gene polymorphisms ( rs12255372 and rs7903146 ) are associated with an increased susceptibility to [breast cancer]{.disease_color} . | 
| [D]{.disease_color}a[G]{.gene_color}       | skin cancer            | COX2        | 1                           | 0.935                           | 73           | Novel                | elevated expression of [cox-2]{.gene_color} has been associated with tumor progression in [skin cancer]{.disease_color} through multiple mechanisms . | 
| [D]{.disease_color}a[G]{.gene_color}       | thyroid cancer         | VEGFA       | 1                           | 0.933                           | 592          | Novel                | as a conclusion , we suggest that [vegf]{.gene_color} g +405 c polymorphism is associated with increased risk of [ptc]{.disease_color} .  | 
| [D]{.disease_color}a[G]{.gene_color}       | stomach cancer         | EGFR        | 1                           | 0.933                           | 1237         | Existing             | recently , high lymph node ratio is closely associated with [egfr]{.gene_color} expression in advanced [gastric cancer]{.disease_color} . | 
| [D]{.disease_color}a[G]{.gene_color}       | liver cancer           | GPC3        | 1                           | 0.933                           | 1944         | Novel                | conclusions serum [gpc3]{.gene_color} was overexpressed in [hcc]{.disease_color} patients .  | 
| [D]{.disease_color}a[G]{.gene_color}       | stomach cancer         | CCR6        | 1                           | 0.931                           | 24           | Novel                | the cox regression analysis showed that high expression of [ccr6]{.gene_color} was an independent prognostic factor for [gc]{.disease_color} patients . | 
| [C]{.compound_color}t[D]{.disease_color}       | Sorafenib    | liver cancer           | 1                           | 0.99                            | 6672         | Existing             | tace plus [sorafenib]{.compound_color} for the treatment of [hepatocellular carcinoma]{.disease_color} : final results of the multicenter socrates trial . | 
| [C]{.compound_color}t[D]{.disease_color}       | Methotrexate | rheumatoid arthritis   | 1                           | 0.989                           | 14546        | Existing             | comparison of low-dose oral pulse [methotrexate]{.compound_color} and placebo in the treatment of [rheumatoid arthritis]{.disease_color} .| 
| [C]{.compound_color}t[D]{.disease_color}       | Auranofin    | rheumatoid arthritis   | 1                           | 0.988                           | 419          | Existing             | [auranofin]{.compound_color} versus placebo in the treatment of [rheumatoid arthritis]{.disease_color} . | 
| [C]{.compound_color}t[D]{.disease_color}       | Lamivudine   | hepatitis B            | 1                           | 0.988                           | 6709         | Existing             | randomized controlled trials ( rcts ) comparing etv with [lam]{.compound_color} for the treatment of [hepatitis b]{.disease_color} decompensated cirrhosis were included .| 
| [C]{.compound_color}t[D]{.disease_color}       | Doxorubicin  | urinary bladder cancer | 1                           | 0.988                           | 930          | Existing             | 17-year follow-up of a randomized prospective controlled trial of adjuvant intravesical [doxorubicin]{.compound_color} in the treatment of superficial [bladder cancer]{.disease_color} . | 
| [C]{.compound_color}t[D]{.disease_color}       | Docetaxel    | breast cancer          | 1                           | 0.987                           | 5206         | Existing             | currently , randomized phase iii trials have demonstrated that [docetaxel]{.compound_color} is an effective strategy in the adjuvant treatment of [breast cancer]{.disease_color} .| 
| [C]{.compound_color}t[D]{.disease_color}       | Cimetidine   | psoriasis              | 0.999                       | 0.987                           | 12           | Novel                | [cimetidine]{.compound_color} versus placebo in the treatment of [psoriasis]{.disease_color} . | 
| [C]{.compound_color}t[D]{.disease_color}       | Olanzapine   | schizophrenia          | 1                           | 0.987                           | 3324         | Novel                | a double-blind , randomised comparative trial of amisulpride versus [olanzapine]{.compound_color} in the treatment of [schizophrenia]{.disease_color} : short-term results at two months . | 
| [C]{.compound_color}t[D]{.disease_color}       | Fulvestrant  | breast cancer          | 1                           | 0.987                           | 826          | Existing             | phase iii clinical trials have demonstrated the clinical benefit of [fulvestrant]{.compound_color} in the endocrine treatment of [breast cancer]{.disease_color} . | 
| [C]{.compound_color}t[D]{.disease_color}       | Pimecrolimus | atopic dermatitis      | 1                           | 0.987                           | 531          | Existing             | introduction : although several controlled clinical trials have demonstrated the efficacy and good tolerability of 1 % [pimecrolimus]{.compound_color} cream for the treatment of [atopic dermatitis]{.disease_color} , the results of these trials may not apply to real-life usage . | 
| [C]{.compound_color}b[G]{.gene_color}       | Gefitinib     | EGFR        | 1                           | 0.99                            | 8746         | Existing             | morphologic features of adenocarcinoma of the lung predictive of response to the [epidermal growth factor receptor]{.gene_color} kinase inhibitors erlotinib and [gefitinib]{.compound_color} .  | 
| [C]{.compound_color}b[G]{.gene_color}       | Adenosine     | EGFR        | 1                           | 0.987                           | 644          | Novel                | it is well established that inhibiting [atp]{.compound_color} binding within the [egfr]{.gene_color} kinase domain regulates its function .   | 
| [C]{.compound_color}b[G]{.gene_color}       | Rosiglitazone | PPARG       | 1                           | 0.987                           | 1498         | Existing             | [rosiglitazone]{.compound_color} is a potent [peroxisome proliferator-activated receptor gamma]{.gene_color} agonist that decreases hyperglycemia by reducing insulin resistance in patients with type 2 diabetes mellitus . | 
| [C]{.compound_color}b[G]{.gene_color}       | D-Tyrosine    | INSR        | 0.998                       | 0.987                           | 1713         | Novel                | this result suggests that [tyrosine]{.compound_color} phosphorylation of phosphatidylinositol 3-kinase by the [insulin receptor]{.gene_color} kinase may increase the specific activity of the former enzyme in vivo .   | 
| [C]{.compound_color}b[G]{.gene_color}       | D-Tyrosine    | IGF1        | 0.998                       | 0.983                           | 819          | Novel                | affinity-purified [insulin-like growth factor i]{.gene_color} receptor kinase is activated by [tyrosine]{.compound_color} phosphorylation of its beta subunit .  | 
| [C]{.compound_color}b[G]{.gene_color}       | Pindolol      | HTR1A       | 1                           | 0.983                           | 175          | Existing             | [pindolol]{.compound_color} , a betablocker with weak partial [5-ht1a receptor]{.gene_color} agonist activity has been shown to produce a more rapid onset of antidepressant action of ssris .   | 
| [C]{.compound_color}b[G]{.gene_color}       | Progesterone  | SHBG        | 1                           | 0.981                           | 492          | Existing             | however , dng also elicits properties of [progesterone]{.compound_color} derivatives like neutrality in metabolic and cardiovascular system and considerable antiandrogenic activity , the latter increased by lack of binding to [shbg]{.gene_color} as specific property of dng . | 
| [C]{.compound_color}b[G]{.gene_color}       | Mifepristone  | AR          | 1                           | 0.98                            | 78           | Existing             | [ru486]{.compound_color} bound to the [androgen receptor]{.gene_color} .  | 
| [C]{.compound_color}b[G]{.gene_color}       | Alfentanil    | OPRM1       | 1                           | 0.979                           | 10           | Existing             | purpose : [alfentanil]{.compound_color} is a high potency [mu opiate receptor]{.gene_color} agonist commonly used during presurgical induction of anesthesia .  | 
| [C]{.compound_color}b[G]{.gene_color}       | Candesartan   | AGTR1       | 1                           | 0.979                           | 36           | Existing             | [tcv-116]{.compound_color} is a new , nonpeptide , [angiotensin ii type-1 receptor]{.gene_color} antagonist that acts as a specific inhibitor of the renin-angiotensin system .  | 
| [G]{.gene_color}i[G]{.gene_color}       | BRCA2       | BRCA1       | 0.972                       | 0.984                           | 12257        | Novel                | a total of 9 families ( 16 % ) showed mutations in the [brca1]{.gene_color} gene , including the one new mutation identified in this study ( 5382insc ) , and 12 families ( 21 % ) presented mutations in the [brca2]{.gene_color} gene . | 
| [G]{.gene_color}i[G]{.gene_color}       | MDM2        | TP53        | 0.938                       | 0.978                           | 17128        | Existing             | no mutations in the [tp53]{.gene_color} gene have been found in samples with amplification of [mdm2]{.gene_color} .  | 
| [G]{.gene_color}i[G]{.gene_color}       | BRCA1       | BRCA2       | 1                           | 0.978                           | 12257        | Existing             | pathogenic truncating mutations in the [brca1]{.gene_color} gene were found in two tumor samples with allelic losses , whereas no mutations were identified in the [brca2]{.gene_color} gene . | 
| [G]{.gene_color}i[G]{.gene_color}       | KRAS        | TP53        | 0.992                       | 0.971                           | 4106         | Novel                | mutations in the [p53]{.gene_color} gene did not correlate with mutations in the [c-k-ras]{.gene_color} gene , indicating that colorectal cancer can develop through pathways independent not only of the presence of mutations in any of these genes but also of their cooperation . | 
| [G]{.gene_color}i[G]{.gene_color}       | TP53        | HRAS        | 0.992                       | 0.969                           | 451          | Novel                | pathologic examination of the uc specimens from aa-exposed patients identified heterozygous [hras]{.gene_color} changes in 3 cases , and deletion or replacement mutations in the [tp53]{.gene_color} gene in 4 . | 
| [G]{.gene_color}i[G]{.gene_color}       | REN         | NR1H3       | 0.998                       | 0.966                           | 8            | Novel                | nuclear receptor [lxralpha]{.gene_color} is involved in camp-mediated human [renin]{.gene_color} gene expression . | 
| [G]{.gene_color}i[G]{.gene_color}       | ESR2        | CYP19A1     | 0.999                       | 0.96                            | 159          | Novel                | dna methylation , histone modifications , and binding of estrogen receptor , [erb]{.gene_color} to regulatory dna sequences of [cyp19a1]{.gene_color} gene were evaluated by chromatin immunoprecipitation ( chip ) assay . | 
| [G]{.gene_color}i[G]{.gene_color}       | RET         | EDNRB       | 0.816                       | 0.96                            | 136          | Novel                | mutations in the [ret]{.gene_color} gene , which codes for a receptor tyrosine kinase , and in [ednrb]{.gene_color} which codes for the endothelin-b receptor , have been shown to be associated with hscr in humans .  | 
| [G]{.gene_color}i[G]{.gene_color}       | PKD1        | PKD2        | 1                           | 0.959                           | 1614         | Existing             | approximately 85 % of adpkd cases are caused by mutations in the [pkd1]{.gene_color} gene , while mutations in the [pkd2]{.gene_color} gene account for the remaining 15 % of cases . | 
| [G]{.gene_color}i[G]{.gene_color}       | LYZ         | CTCF        | 0.999                       | 0.959                           | 2            | Novel                | in conjunction with the thyroid receptor ( tr ) , [ctcf]{.gene_color} binding to the [lysozyme]{.gene_color} gene transcriptional silencer mediates the thyroid hormone response element ( tre ) - dependent transcriptional repression . |  
Table: Contains the top ten predictions for each edge type. Highlighted words represent entities mentioned within the given sentence. {#tbl:edge_prediction_tbl}

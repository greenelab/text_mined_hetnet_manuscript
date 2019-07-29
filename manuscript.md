---
author-meta:
- David N. Nicholson
- Daniel S. Himmelstein
- Casey S. Greene
date-meta: '2019-07-29'
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
([permalink](https://greenelab.github.io/text_mined_hetnet_manuscript/v/77307b470aa2d92f7d31b4a5eebc6658154a5dc7/))
was automatically generated
from [greenelab/text_mined_hetnet_manuscript@77307b4](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/77307b470aa2d92f7d31b4a5eebc6658154a5dc7)
on July 29, 2019.
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

Knowledge bases support multiple research efforts including providing contextual information for biomedical entities, constructing networks, and supporting the interpretation of high-throughput analyses.
Some knowledge bases are automatically constructed, but most are populated via some form of manual curation.
Manual curation is time consuming and difficult to scale in the context of an increasing publication rate.
A recently described "data programming" paradigm seeks to circumvent this arduous process by combining distant supervision with simple rules and heuristics written as labeling functions that can be automatically applied to inputs.
Unfortunately writing useful label functions requires substantial error analysis and is a non trivial task: in early efforts to use data programming we found that producing each label function could take a few days.
Producing a biomedical knowledge base with multiple node and edge types could take hundreds or thousands of label functions.
In this paper we sought to evaluate the extent to which label functions could be re-used across edge types. 
We used a subset of Hetionet v1 that centered on disease, compound, and gene nodes to evaluate this approach.
We compare a baseline distant supervision model with the same distant supervision resources added to edge-type-specific label functions, edge-type-mismatch label functions, and all label functions.
We confirmed that adding additional edge-type-specific label functions improves performance.
We also found that adding one or a few edge-type-mismatch label functions also nearly always improves performance.
Adding a large number of edge-type-mismatch label functions produces more variable performance that depends on the edge type being predicted and the edge type that is the source of the label function.
Lastly, we show that this approach, even on this subgraph of Hetionet, could certain novel edges to Hetionet v1 with high confidence.
We expect that its use in practice would include additional filtering and scoring methods which would further enhance precision.



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

## Materials and Methods

### Hetionet

Hetionet [@O21tn8vf] is a large heterogenous network that contains pharmacological and biological information.
This network depicts information in the form of nodes and edges of different types: nodes that represent biological and pharmacological entities and edges which represent relationships between entities. 
Hetionet v1.0 contains 47,031 nodes with 11 different data types and 2,250,197 edges that represent 24 different relationship types (Figure {@fig:hetionet}).
Edges in Hetionet were obtained from open databases, such as the GWAS Catalog [@16cIDAXhG] and DrugBank [@1FI8iuYiQ].
For this project, we analyzed performance over a subset of the Hetionet relationship types: disease associates with a gene (DaG), compound binds to a gene (CbG), gene interacts with gene (GiG) and compound treating a disease (CtD).

![
A metagraph (schema) of Hetionet where pharmacological, biological and disease entities are represented as nodes and the relationships between them are represented as edges.
This project only focuses on the information shown in bold; however, we can extend this work to incorporate the faded out information as well.
](images/figures/hetionet/metagraph_highlighted_edges.png){#fig:hetionet}

### Dataset

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
We hand labeled five hundred to a thousand candidate sentences of each relationship to obtain to obtain a ground truth set (Table {@tbl:candidate-sentences}, [dataset](https://github.com/greenelab/text_mined_hetnet_manuscript/tree/master/supplementary_materials/annotated_sentences)).

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

### Training Models

#### Generative Model

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

#### Word Embeddings

Word embeddings are representations that map individual words to real valued vectors of user-specified dimensions.
These embeddings have been shown to capture the semantic and syntactic information between words [@u5iJzbp9].
Using all candidate sentences for each individual relationship pair, we trained facebook's fastText [@qUpCDz2v] to generate word embeddings.
The fastText model uses a skipgram model [@1GhHIDxuW] that aims to predict the context given a candidate word and pairs the model with a novel scoring function that treats each word as a bag of character n-grams.
We trained this model for 20 epochs using a window size of 2 and generated 300-dimensional word embeddings.
We use the optimized word embeddings to train a discriminative model.  

#### Discriminative Model

The discriminative model is a neural network, which we train to predict labels from the generative model.
The expectation is that the discriminative model can learn more complete features of the text than the label functions that are used in the generative model.
We used a convolutional neural network with multiple filters as our discriminative model.
This network uses multiple filters with fixed widths of 300 dimensions and a fixed height of 7 (Figure {@fig:convolutional_network}), because this height provided the best performance in terms of relationship classification [@fs8rAHoJ].
We trained this model for 20 epochs using the adam optimizer [@c6d3lKFX] with a learning rate of 0.001.
This optimizer used pytorch's default parameter settings.
We added a L2 penalty on the network weights to prevent overfitting.
Lastly, we added a dropout layer (p=0.25) between the fully connected layer and the softmax layer.

![
The architecture of the discriminative model is a convolutional neural network.
We perform a convolution step using multiple filters. 
These filters generate a feature map that is sent into a maximum pooling layer. 
This layer extracts the largest feature in each of these maps.
The extracted features are concatenated into a singular vector that is passed into a fully connected network. 
The fully connected network has 300 neurons for the first layer, 100 neurons for the second layer and 50 neurons for the last layer. 
From the fully connected network the last step is to generate predictions using the softmax layer.
](images/figures/convolutional_neural_network/convolutional_neural_nework.png){#fig:convolutional_network}

#### Calibration of the Discriminative Model

Often many tasks require a machine learning model to output reliable probability predictions. 
A model is well calibrated if the probabilities emitted from the model match the observed probabilities: a well-calibrated model that assigns a class label with 80% probability should have that class appear 80% of the time.
Deep neural network models can often be poorly calibrated [@QJ6hYH8N; @rLVjMJ5l].
These models are usually over-confident in their predictions.
As a result, we calibrated our convolutional neural network using temperature scaling. 
Temperature scaling uses a parameter T to scale each value of the logit vector (z) before being passed into the softmax (SM) function.

$$\sigma_{SM}(\frac{z_{i}}{T}) = \frac{\exp(\frac{z_{i}}{T})}{\sum_{i}\exp(\frac{z_{i}}{T})}$$

We found the optimal T by minimizing the negative log likelihood (NLL) of a held out validation set.
The benefit of using this method is the model becomes more reliable and the accuracy of the model doesn't change [@QJ6hYH8N].

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

#### Adding Random Noise to Generative Model

We discovered in the course of this work that adding a single label function from a mismatched type would often improve the performance of the generative model (see Results).
We designed an experiment to test whether adding a noisy label function also increased performance.
This label function emitted a positive or negative label at varying frequencies, which were evenly spaced from zero to one.
Zero is the same as distant supervision alone.
We trained the generative model with these label functions added and report results in terms of AUROC and AUPR.



## Results

### Generative Model Using Randomly Sampled Label Functions
![
Grid of Area Under the Receiver Operating Curve (AUROC) scores for each generative model trained on randomly sampled label functions.
The rows depict the relationship each model is trying to predict and the columns are the relationship specific sources each label function is sampled from.
For example, the top-left most square depicts the generative model predicting Disease associates Gene (DaG) sentences, while randomly sampling label functions designed to predict the DaG relationship. 
The square towards the right depicts the generative model predicting DaG sentences, while randomly sampling label functions designed to predict the Compound treats Disease (CtD) relationship.
This pattern continues filling out the rest of the grid.
The last most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/greenelab/snorkeling/master/figures/label_sampling_experiment/transfer_test_set_auroc.png){#fig:gen_model_auroc}

We added randomly sampled label functions to a baseline for each edge type to evaluate the feasibility of label function re-use.
Our baseline model consisted of a generative model trained with only the edge type's distant supervision label functions.
We report the results in the form of area under the precision recall curve (AUPR) (Figure {@fig:gen_model_auprc}) and area under the receiver operating curve (AUROC) (Figure {@fig:gen_model_auroc}).  
The on-diagonal plots of figure {@fig:gen_model_auprc}) and figure {@fig:gen_model_auprc} show performance when edge-specific label functions are added on top of edge-specific baselines.
The general trend is performance increases in this setting.
The Compound-treats-Disease (CtD) edge type is a quintessential example of this trend.
The baseline model starts off with an AUROC score of 52% and an AUPRC of 28%, which increase to 76% and 49% respectively as more CtD label functions are included. 
Disease-associates-Gene (DaG) edges have a similar trend: performance starting off with a AUROC of 56% and AUPRC of 41%, which increase to 62% and 45% respectively.
Both the Compound-binds-Gene (CbG) and Gene-interacts-Gene (GiG) edges have an increasing trend but plateau after a few label functions are added.  

The off-diagonals in figure {@fig:gen_model_auprc}) and figure {@fig:gen_model_auprc} show how performance varies when label functions from one edge type are added to a different edge type's baseline.
In certain cases (apparent for DaG), performance increases regardless of the edge type used for label functions.
In other cases (apparent with CtD), one label function appears to improve performance; however, adding more label functions does not improve performance (AUROC) or decreases it (AUPRC).
In certain cases, the source of the label functions appear to be important: for CbG edges performance decreases when using label functions from the DaG and CtD categories.

Our initial hypothesis was based on the idea that certain edge types capture similar physical relationships and that these cases would be particularly amenable for label function transfer.
For example, Compound-binds-Gene (CbG) and Gene-interacts-Gene (GiG) both describe physical interactions.
We observed that performance increased as assessed by both AUPRC and AUPRC when using label functions from the GiG edge type to predict CbG edges.
A similar trend was observed when predicting the GiG edge; however, the performance differences were small for this edge type making the importance difficult to assess.  
The last column shows performance when sampling from all label functions.
Performance increased (AUROC and AUPRC) for both DaG and CtD, when sampling from the full pool of label functions.
CbG and GiG also had increased performance when one random label function was sampled, but performance decreased drastically as more label functions were added.
It is possible that a small number of irrelevant label functions are able to overwhelm the distant supervision label functions in these cases.

![
Grid of Area Under the Precision Recall Curve (AUPRC) scores for each generative model trained on randomly sampled label functions.
The rows depict the relationship each model is trying to predict and the columns are the relationship specific sources each label function is sampled from.
For example, the top-left most square depicts the generative model predicting Disease associates Gene (DaG) sentences, while randomly sampling label functions designed to predict the DaG relationship. 
The square towards the right depicts the generative model predicting DaG sentences, while randomly sampling label functions designed to predict the Compound treats Disease (CtD) relationship.
This pattern continues filling out the rest of the grid.
The last most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/greenelab/snorkeling/master/figures/label_sampling_experiment/transfer_test_set_auprc.png){#fig:gen_model_auprc}

### Random Label Function Gen Model Analysis
![
A grid of area under the receiver operating curve (AUROC) for each edge type.
Each plot consists of adding a single label function on top of the baseline model.
This label function emits a positive (top row) or negative (bottom row) label at specified frequencies, and performance at zero is equivalent to not having a randomly emitting label function.
The error bars represent 95% confidence intervals for AUROC (y-axis) at each emission frequency.
](https://raw.githubusercontent.com/danich1/snorkeling/f8962788e462b783be05a6dec5eec7fe0f0259e7/figures/gen_model_error_analysis/transfer_test_set_auroc.png){#fig:random_label_function_auroc}

We observed that including one label function of a mismatched type to distant supervision often improved performance, so we evaluated the effects of adding a random label function in the same setting.
We found that adding random noise did not usually improve performance (Figures {@fig:random_label_function_auprc} and {@fig:random_label_function_auprc}).
For the CbG edge type we did observe slightly increased performance via AUPR (Figure {@fig:random_label_function_auprc}).
However, in general the performance changes were smaller than those observed with mismatched label types.

![
A grid of area under the precision recall curve (AUPR) for each edge type.
Each plot consists of adding a single label function on top of the baseline model.
This label function emits a positive (top row) or negative (bottom row) label at specified frequencies.
The error bars represent 95% confidence intervals for AUPR (y-axis) at emission frequency.
](https://raw.githubusercontent.com/danich1/snorkeling/f8962788e462b783be05a6dec5eec7fe0f0259e7/figures/gen_model_error_analysis/transfer_test_set_auprc.png){#fig:random_label_function_auprc}


### Discriminative Model Performance

In this framework we used a generative model trained over label functions to produce probabilistic training labels for each sentence.
Then we train a discriminative model, which has full access to a representation of the text of the sentence, to predict the generated labels.
The discriminative model is a convolutional neural network trained over word embeddings.
We report the results of the discriminative model using AUPR (Figure {@fig:discriminative_model_auprc}) and AUROC (Figure {@fig:discriminative_model_auroc}).    
We found that the discriminative model under-performed the generative model in most cases.
Only for the CtD edge does the discriminative model appear to provide performance above the generative model and that increased performance is only with modest numbers of label functions.
With the full set of label functions, the performance of both remains similar.
The trend observed in the generative model that one or a few mismatched label functions (off-diagonal) improves performance is retained despite the limited performance of the discriminative model.
 

![
Grid of Area Under the Receiver Operating Curve (AUROC) scores for each discriminative model trained using generated labels from the generative models.
The rows depict the relationship each model is trying to predict and the columns are the relationship specific sources each label function was sampled from. 
For example, the top-left most square depicts the discriminator model predicting Disease associates Gene (DaG) sentences, while randomly sampling label functions designed to predict the DaG relationship.
The error bars over the points represents the standard deviation between sampled runs.
The square towards the right depicts the discriminative model predicting DaG sentences, while randomly sampling label functions designed to predict the Compound treats Disease (CtD) relationship.
This pattern continues filling out the rest of the grid.
The last most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/greenelab/snorkeling/master/figures/label_sampling_experiment/disc_performance_test_set_auroc.png){#fig:discriminative_model_auroc}

![
Grid of Area Under the Receiver Operating Curve (AUROC) scores for each discriminative model trained using generated labels from the generative models.
The rows depict the relationship each model is trying to predict and the columns are the relationship specific sources each label function was sampled from. 
For example, the top-left most square depicts the discriminator model predicting Disease associates Gene (DaG) sentences, while randomly sampling label functions designed to predict the DaG relationship.
The error bars over the points represents the standard deviation between sampled runs.
The square towards the right depicts the discriminative model predicting DaG sentences, while randomly sampling label functions designed to predict the Compound treats Disease (CtD) relationship.
This pattern continues filling out the rest of the grid.
The last most column consists of pooling every relationship specific label function and proceeding as above.
](https://raw.githubusercontent.com/greenelab/snorkeling/master/figures/label_sampling_experiment/disc_performance_test_set_auprc.png){#fig:discriminative_model_auprc}

### Discriminative Model Calibration

Even deep learning models with high precision and recall can be poorly calibrated, and the overconfidence of these models has been noted [@QJ6hYH8N; @rLVjMJ5l].
We attempted to calibrate the best performing discriminative model so that we could directly use the emitted probabilities.
We examined the calibration of our existing model (Figure {@fig:discriminative_model_calibration}, blue line).
We found that the DaG and CtG edge types were, though not perfectly calibrated, were somewhat aligned with the ideal calibration lines.
The CbG and GiG edges were poorly calibrated and increasing model certainty did not always lead to an increase in precision.
Applying the calibration algorithm (orange line) did not appear to bring predictions in line with the ideal calibration line, but did capture some of the uncertainty in the GiG edge type.
For this reason we use the measured precision instead of the predicted probabilities when determining how many edges could be added to existing knowledge bases with specified levels of confidence.

![
Calibration plots for the discriminative model.
A perfectly calibrated model would follow the dashed diagonal line.
The blue line represents the predictions before calibration and the orange line shows predictions after calibration. 
](https://raw.githubusercontent.com/greenelab/snorkeling/master/figures/model_calibration_experiment/model_calibration.png){#fig:discriminative_model_calibration}

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

<style> span.gene_color { color:#02b3e4 } span.disease_color { color:#875442 } span.compound_color { color:#e91e63 } </style> 
## Supplemental Figures {.page_break_before}

### Confidence Scores for Each Edge Type 

| Disease Name                 | Gene Symbol | Text                                                                                                                                                                                                                                                                                                                                      | Before Calibration | After Calibraiton | 
|------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| adrenal gland cancer         | TP53        | the mechanisms of adrenal tumorigenesis remain poorly established ; the r337h germline mutation in the [p53]{.gene_color} gene has previously been associated with [acts]{.disease_color} in brazilian children .                                                                                                                                  | 1.0                | 0.882             | 
| breast cancer                | ERBB2       | in [breast cancer]{.disease_color} , overexpression of [her2]{.gene_color} is associated with an aggressive tumor phenotype and poor prognosis .                                                                                                                                                                                                   | 1.0                | 0.845             | 
| lung cancer                  | TP53        | however , both adenine ( a ) and guanine ( g ) mutations are found in the [p53]{.gene_color} gene in cr exposure-related [lung cancer]{.disease_color} .                                                                                                                                                                                           | 1.0                | 0.83              | 
| malignant glioma             | BAX         | these data suggest that the combination of tra-8 treatment with specific overexpression of [bax]{.gene_color} using advegfbax may be an effective approach for the treatment of human [malignant gliomas]{.disease_color} .                                                                                                                        | 0.999              | 0.827             | 
| polycystic ovary syndrome    | SERPINE1    | 4 g allele in [pai-1]{.gene_color} gene was more frequent in [pcos]{.disease_color} and the 4g/4 g genotype was associated with increased pai-1 levels .                                                                                                                                                                                           | 0.999              | 0.814             | 
| systemic lupus erythematosus | PRL         | results : [sle]{.disease_color} patients showed a significantly higher serum level of [prl]{.gene_color} than healthy subjects , which was especially obvious in the active stage of the disease ( p = 0.000 .                                                                                                                                     | 0.999              | 0.813             | 
| hematologic cancer           | TNF         | the mean [tnf-alpha]{.gene_color} plasma concentration in the patients with [cll]{.disease_color} was significantly higher than in the healthy control population ( 16.4 versus 8.7 pg/ml ; p < .0001 ) .                                                                                                                                          | 0.999              | 0.81              | 
| lung cancer                  | MUC16       | the mean concentration of [ca 125]{.gene_color} was higher in patients with [lung cancer]{.disease_color} ( 37 + / - 81 u/ml ) than in those with nonmalignant disease ( 4.2 + / - 5.7 u/ml ) ( p less than 0.01 ) .                                                                                                                               | 0.999              | 0.806             | 
| prostate cancer              | AR          | the [androgen receptor]{.gene_color} was expressed in all primary and metastatic [prostate cancer]{.disease_color} tissues and no mutations were identified .                                                                                                                                                                                      | 0.999              | 0.801             | 
| breast cancer                | ERBB2       | the results of multiple linear regression analysis , with her2 as the dependent variable , showed that family history of [breast cancer]{.disease_color} was significantly associated with elevated [her2]{.gene_color} levels in the tumors ( p = 0.0038 ) , after controlling for the effects of age , tumor estrogen receptor , and dna index . | 0.999              | 0.8               |  
Table: Contains the top ten Disease-associates-Gene confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Gene mentions are highlighted in [blue]{.gene_color}.{#tbl:dg_top_ten_table}

| Disease Name       | Gene Symbol | Text                                                                                                                                                                                                                                                                                                                                                           | Before Calibration | After Calibraiton | 
|--------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| breast cancer      | NAT2        | [ the relationship between passive smoking , [breast cancer]{.disease_color} risk and [n-acetyltransferase 2]{.gene_color} ( nat2 ) ] .                                                                                                                                                                                                                                 | 0.012              | 0.287             | 
| schizophrenia      | EP300       | ventricle size and [p300]{.gene_color} in [schizophrenia]{.disease_color} .                                                                                                                                                                                                                                                                                             | 0.012              | 0.286             | 
| hematologic cancer | CD33        | in the 2 ( nd ) study of [cd33]{.gene_color} + [sr-aml]{.disease_color} 2 doses of go ( 4.5 - 9 mg/m ( 2 ) ) were administered > = 60d post reduced intensity conditioning ( ric ) allosct ( 8 wks apart ) .                                                                                                                                                            | 0.01               | 0.281             | 
| Crohn's disease    | PTPN2       | in this sample , we were able to confirm an association between [cd]{.disease_color} and [ptpn2]{.gene_color} ( genotypic p = 0.019 and allelic p = 0.011 ) , and phenotypic analysis showed an association of this snp with late age at first diagnosis , inflammatory and penetrating cd behaviour , requirement of bowel resection and being a smoker at diagnosis . | 0.008              | 0.268             | 
| breast cancer      | ERBB2       | [ long-term efficacy and safety of adjuvant trastuzumab for [her2-positive]{.gene_color} early [breast cancer ]]{.disease_color} .                                                                                                                                                                                                                                      | 0.007              | 0.262             | 
| hematologic cancer | CD40LG      | we examined the direct effect of lenalidomide on [cll-cell proliferation]{.disease_color} induced by [cd154-expressing]{.gene_color} accessory cells in media containing interleukin-4 and -10 .                                                                                                                                                                        | 0.006              | 0.259             | 
| hematologic cancer | MLANA       | methods : the sln sections ( n = 214 ) were assessed by qrt assay for 4 established messenger rna biomarkers : [mart-1]{.gene_color} , mage-a3 , [galnac-t]{.disease_color} , and pax3 .                                                                                                                                                                                | 0.005              | 0.252             | 
| breast cancer      | ERBB2       | the keywords erbb2 or her2 or erbb-2 or [her-2]{.gene_color} and [breast cancer]{.disease_color} and ( country ) were used to search pubmed , international and local conference abstracts and local-language journals from the year 2000 onwards .                                                                                                                     | 0.003              | 0.225             | 
| hepatitis B        | PKD2        | conversely , a significant enhancement of activation was observed for afb1 in cases of mild cah and especially for [trp-p-2]{.gene_color} in [hepatitis b]{.disease_color} virus carriers , irrespective of their histologic diagnosis .                                                                                                                                | 0.002              | 0.217             | 
| hematologic cancer | C7          | serum antibody responses to four haemophilus influenzae type b capsular polysaccharide-protein conjugate vaccines ( prp-d , hboc , [c7p]{.gene_color} , and [prp-t )]{.disease_color} were studied and compared in 175 infants , 85 adults and 140 2-year-old children .                                                                                                | 0.002              | 0.208             | 
Table: Contains the bottom ten Disease-associates-Gene confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Gene mentions are highlighted in [blue]{.gene_color}. {#tbl:dg_bottom_ten_table}

| Compound Name      | Disease Name                   | Text                                                                                                                                                                       | Before Calibration | After Calibration | 
|--------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| Methylprednisolone | asthma                         | use of tao without [methylprednisolone]{.compound_color} in the treatment of severe [asthma]{.disease_color} .                                                                          | 1.0                | 0.895             | 
| Methyldopa         | hypertension                   | atenolol and [methyldopa]{.compound_color} in the treatment of [hypertension]{.disease_color} .                                                                                         | 1.0                | 0.888             | 
| Prednisone         | asthma                         | [prednisone]{.compound_color} and beclomethasone for treatment of [asthma]{.disease_color} .                                                                                            | 1.0                | 0.885             | 
| Prazosin           | hypertension                   | experience with [prazosin]{.compound_color} in the treatment of [hypertension]{.disease_color} .                                                                                        | 1.0                | 0.883             | 
| Prazosin           | hypertension                   | [prazosin]{.compound_color} in the treatment of [hypertension]{.disease_color} .                                                                                                        | 1.0                | 0.878             | 
| Prazosin           | hypertension                   | [ [prazosin]{.compound_color} in the treatment of [hypertension ]]{.disease_color} .                                                                                                    | 1.0                | 0.878             | 
| Methyldopa         | hypertension                   | oxprenolol plus cyclopenthiazide-kcl versus [methyldopa]{.compound_color} in the treatment of [hypertension]{.disease_color} .                                                          | 1.0                | 0.877             | 
| Prednisolone       | lymphatic system cancer        | peptichemio : a new oncolytic drug in combination with vincristine and [prednisolone]{.compound_color} in the treatment of [non-hodgkin lymphomas]{.disease_color} .                    | 1.0                | 0.871             | 
| Methyldopa         | hypertension                   | methyldopate , the ethyl ester hydrochloride salt of [alpha-methyldopa]{.compound_color} ( alpha-md ) , is used extensively in the treatment of severe [hypertension]{.disease_color} . | 1.0                | 0.851             | 
| Haloperidol        | Gilles de la Tourette syndrome | a comparison of pimozide and [haloperidol]{.compound_color} in the treatment of gilles de la [tourette 's syndrome]{.disease_color} .                                                   | 1.0                | 0.839             | 
Table: Contains the top ten Compound-treats-Disease confidence scores after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Compound mentions are highlighted in [red]{.compound_color}.{#tbl:cd_top_ten_table}

| Compound Name                  | Disease Name            | Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Before Calibration | After Calibration | 
|--------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| Dexamethasone                  | hypertension            | [dexamethasone]{.compound_color} and [hypertension]{.disease_color} in preterm infants .                                                                                                                                                                                                                                                                                                                                                                                                              | 0.011              | 0.34              | 
| Reserpine                      | hypertension            | [reserpine]{.compound_color} in [hypertension]{.disease_color} : present status .                                                                                                                                                                                                                                                                                                                                                                                                                     | 0.01               | 0.336             | 
| Creatine                       | coronary artery disease | scintiphotographic findings were compared with the size of [myocardial infarcts]{.disease_color} calculated from measurements of the activity of mb isoenzymes of [creatine]{.compound_color} kinase ( ck-mb ) in serum and in the myocardium at autopsy , as described by sobel 's method .                                                                                                                                                                                                          | 0.009              | 0.334             | 
| Hydrocortisone                 | brain cancer            | to explore the effects of repeated episodes of hypercortisolemia on [hypothalamic-pituitary-adrenal axis]{.disease_color} regulation , we studied plasma acth and [cortisol]{.compound_color} ( cort ) responses to 100 micrograms human crh ( hcrh ) in 10 dexamethasone ( 1.5 mg ) - pretreated elderly endurance athletes who had abstained from physical activity for at least 48 h before testing and 13 sedentary age-matched controls .                                                        | 0.009              | 0.333             | 
| Hydrocortisone                 | brain cancer            | basal activity of the [hypothalamic-pituitary-adrenal axis]{.disease_color} was estimated by determinations of 24-h urinary free cortisol-excretion , evening basal plasma total and free [cortisol]{.compound_color} concentrations , and the cortisol binding globulin-binding capacity .                                                                                                                                                                                                           | 0.008              | 0.328             | 
| Creatine                       | coronary artery disease | during successful and uncomplicated angioplasty ( ptca ) , we studied the effect of a short lasting [myocardial ischemia]{.disease_color} on plasma creatine kinase , creatine kinase mb-activity , and [creatine]{.compound_color} kinase mm-isoforms ( mm1 , mm2 , mm3 ) in 23 patients .                                                                                                                                                                                                           | 0.006              | 0.318             | 
| Benzylpenicillin               | epilepsy syndrome       | it was shown in experiments on cats under nembutal anesthesia that a lesion of the medial forebrain bundle ( mfb ) and partly of the preoptic region at the side of local penicillin application on the cerebral cortex ( g. suprasylvius medius ) results in depression of the [epileptiform activity]{.disease_color} in the [penicillin-induced]{.compound_color} focus , as well as in the secondary `` mirror '' focus , which appeared in the symmetrical cortex area of the other hemisphere . | 0.005              | 0.315             | 
| Indomethacin                   | hypertension            | effects of [indomethacin]{.compound_color} in rabbit [renovascular hypertension]{.disease_color} .                                                                                                                                                                                                                                                                                                                                                                                                    | 0.004              | 0.308             | 
| Cyclic Adenosine Monophosphate | ovarian cancer          | the hormonal regulation of steroidogenesis and [adenosine 3 ' :5 ' - cyclic monophosphate]{.compound_color} in [embryonic-chick ovary]{.disease_color} .                                                                                                                                                                                                                                                                                                                                              | 0.002              | 0.292             | 
| Dobutamine                     | coronary artery disease | two-dimensional echocardiography can detect regional wall motion abnormalities resulting from [myocardial ischemia]{.disease_color} produced by [dobutamine]{.compound_color} infusion .                                                                                                                                                                                                                                                                                                              | 0.002              | 0.287             |  
Table: Contains the bottom ten Compound-treats-Disease confidence scores before and after model calbration. Disease mentions are highlighted in [brown]{.disease_color} and Compound mentions are highlighted in [red]{.compound_color}.{#tbl:cd_bottom_ten_table}

| Compound Name   | Gene Symbol | Text                                                                                                                                                                                                                                                                                                                                                                                                         | Before Calibration | After Calibration | 
|-----------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| Hydrocortisone  | SHBG        | serum concentrations of testicular and adrenal androgens and androgen precursors , [cortisol]{.compound_color} , unconjugated ( e1 ) and total estrone ( te1 ; greater than or equal to 85 % e1 sulfate ) , pituitary hormones , sex hormone binding globulin ( [shbg )]{.gene_color} and albumin were measured in 14 male patients with non-diabetic end stage renal disease and in 28 age-matched healthy controls . | 0.997              | 0.745             | 
| Minoxidil       | EGFR        | direct measurement of the ability of minoxidil to compete for binding to the egf receptor indicated that [minoxidil]{.compound_color} probably does not bind to the [egf receptor]{.gene_color} .                                                                                                                                                                                                                      | 0.99               | 0.706             | 
| Hydrocortisone  | SHBG        | gonadotropin , testosterone , sex hormone binding globulin ( [shbg )]{.gene_color} , dehydroepiandrosterone sulphate , androstenedione , estradiol , prolactin , [cortisol]{.compound_color} , thyrotropin , and free thyroxine levels were determined .                                                                                                                                                               | 0.988              | 0.7               | 
| Cholecalciferol | DBP         | [cholecalciferol]{.compound_color} ( vitamin d3 ) and its 25-hydroxy metabolite are transported in plasma bound to a specific protein , the binding protein for cholecalciferol and its metabolites ( [dbp )]{.gene_color} .                                                                                                                                                                                           | 0.983              | 0.685             | 
| Indomethacin    | AGT         | [indomethacin]{.compound_color} , a potent inhibitor of prostaglandin synthesis , is known to increase the maternal blood pressure response to [angiotensin ii]{.gene_color} infusion .                                                                                                                                                                                                                                | 0.982              | 0.68              | 
| Tretinoin       | RXRA        | the vitamin a derivative [retinoic acid]{.compound_color} exerts its effects on transcription through two distinct classes of nuclear receptors , the retinoic acid receptor ( rar ) and the [retinoid x receptor]{.gene_color} ( rxr ) .                                                                                                                                                                              | 0.975              | 0.668             | 
| Dopamine        | NTS         | [neurotensin]{.gene_color} binding was not modified by the addition of [dopamine]{.compound_color} .                                                                                                                                                                                                                                                                                                                   | 0.97               | 0.659             | 
| D-Tyrosine      | PLCG1       | epidermal growth factor ( egf ) or platelet-derived growth factor binding to their receptor on fibroblasts induces tyrosine phosphorylation of plc gamma 1 and stable association of [plc gamma 1]{.gene_color} with the receptor protein [tyrosine]{.compound_color} kinase .                                                                                                                                         | 0.969              | 0.659             | 
| D-Tyrosine      | PLCG1       | [tyrosine]{.compound_color} phosphorylation of plc-ii was stimulated by low physiological concentrations of egf ( 1 nm ) , was quantitative , and was already maximal after a 30 sec incubation with 50 nm egf at 37 degrees c. interestingly , antibodies specific for plc-ii were able to coimmunoprecipitate the egf receptor and antibodies against egf receptor also coimmunoprecipitated [plc-ii]{.gene_color} . | 0.964              | 0.651             | 
| Ketamine        | C5          | additionally , reduction of glycine binding by the c-5 antagonists was reversed by both nmda receptor agonists and c-7 competitive [nmda]{.compound_color} antagonists , providing evidence that the site of action of these [c-5]{.gene_color} antagonists is the nmda recognition site , resulting in indirect modulation of the glycine site .                                                                      | 0.957              | 0.643             | 
Table: Contains the top ten Compound-treats-Disease confidence scores before and after model calbration. Gene mentions are highlighted in [blue]{.gene_color} and Compound mentions are highlighted in [red]{.compound_color}.{#tbl:cg_top_ten_table}

| Compound Name  | Gene Symbol | Text                                                                                                                                                                                                                                                                                                                                                                                            | Before Calibration | After Calibration | 
|----------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| Iron           | NDUFB3      | since gastric acid plays an important role in the absorption process of [iron]{.compound_color} and vitamin b12 , we determined levels of iron , ferritin , vitamin [b12]{.gene_color} , and folic acid in 75 serum samples obtained during continuous omeprazole therapy ( 6-48 months after start of therapy ) from 34 patients with peptic diseases ( primarily reflux esophagitis ) .                 | 0.006              | 0.276             | 
| D-Tyrosine     | PLAU        | either the 55 kda u-pa form and the lower mw form ( 33 kda ) derived from the 55 kda [u-pa]{.gene_color} are tyr-phosphorylated also the u-pa secreted in the culture media of human fibrosarcoma cells ( ht-1080 ) is phosphorylated in [tyrosine]{.compound_color} as well as u-pa present in tissue extracts of tumors induced in nude mice by ht-1080 cells .                                         | 0.006              | 0.276             | 
| D-Leucine      | POMC        | cross-reactivities of [leucine-enkephalin]{.compound_color} and [beta-endorphin]{.gene_color} with the eia were less than 0.1 % , while that with gly-gly-phe-met and oxidized gly-gly-phe-met were 2.5 % and 10.2 % , respectively .                                                                                                                                                                     | 0.006              | 0.273             | 
| Eprazinone     | GAST        | in patients with renal failure there exists the inhibition of the gastrin acid secretion which is the cause of the weakening of the mechanism of the feedback connection between [hcl]{.compound_color} and [gastrin]{.gene_color} , while because of a permanent stimulation of g-cells , the hyperplasia of these cells develops , as well as the increased secretory activity , and hypergastrinemia . | 0.005              | 0.271             | 
| Hydrocortisone | GH1         | luteinizing hormone responses to luteinizing hormone releasing hormone , and [growth hormone]{.gene_color} and [cortisol]{.compound_color} responses to insulin induced hypoglycaemia in functional secondary amenorrhoea .                                                                                                                                                                               | 0.005              | 0.271             | 
| Hydrocortisone | GH1         | group iv patients had normal basal levels of lh and normal lh , [gh]{.gene_color} and [cortisol]{.compound_color} responses .                                                                                                                                                                                                                                                                             | 0.005              | 0.269             | 
| Bupivacaine    | AVP         | plasma renin activity and [vasopressin]{.gene_color} concentration , arterial pressure , and serum osmolality were measured in 17 patients before and after random epidural injection of either 6.7 ml of 0.75 % [bupivacaine]{.compound_color} ( n = 7 ) or the same volume of saline ( n = 10 ) .                                                                                                       | 0.004              | 0.26              | 
| Epinephrine    | INS         | thermogenic effect of thyroid hormones : interactions with [epinephrine]{.compound_color} and [insulin]{.gene_color} .                                                                                                                                                                                                                                                                                    | 0.004              | 0.259             | 
| Hydrocortisone | GH1         | [cortisol]{.compound_color} and [growth hormone]{.gene_color} ( gh ) secretion ( spontaneous variations at night and the release induced by insulin hypoglycaemia ) were investigated in 69 children and adolescents .                                                                                                                                                                                    | 0.002              | 0.241             | 
| Estriol        | LGALS1      | [ diagnostic value of serial determination of [estriol]{.compound_color} and [hpl]{.gene_color} in plasma and of total estrogens in 24-h-urine compared to single values for diagnosis of fetal danger ] .                                                                                                                                                                                                | 0.0                | 0.181             |
Table: Contains the bottom ten Compound-binds-Gene confidence scores before and after model calbration. Gene mentions are highlighted in [blue]{.gene_color} and Compound mentions are highlighted in [red]{.compound_color}.{#tbl:cg_bottom_ten_table}

| Gene1 Symbol | Gene2 Symbol | Text                                                                                                                                                                                                                                                         | Before Calibration | After Calibration | 
|--------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| INS          | HSPA4        | conclusions : intact insulin only weakly interacts with the [hsp70]{.gene_color} chaperone dnak whereas monomeric proinsulin and peptides from 3 distinct [proinsulin]{.gene_color} regions show substantial chaperone binding .                                   | 0.834              | 0.574             | 
| NMT1         | S100B        | values for k ( cat ) indicated that , once gag or [nef]{.gene_color} binds to the enzyme , myristoylation by [nmt1]{.gene_color} and nmt2 proceeds at comparable rates .                                                                                           | 0.826              | 0.571             | 
| VEGFA        | HIF1A        | mechanistically , we demonstrated that resveratrol inhibited [hif-1alpha]{.gene_color} and [vegf]{.gene_color} expression through multiple mechanisms .                                                                                                            | 0.82               | 0.569             | 
| ITGAV        | PECAM1       | antigens expressed on emp and ec were assayed flow cytometrically and included constitutive markers ( [cd31]{.gene_color} , [cd51/61]{.gene_color} , cd105 ) , inducible markers ( cd54 , cd62e and cd106 ) , and annexin v binding .                              | 0.81               | 0.566             | 
| F10          | PF4          | these compounds inhibit both [factor xa]{.gene_color} and thrombin , in the presence of antithrombin , while they are devoid of undesirable non-specific interactions , particularly with [platelet factor 4]{.gene_color} ( pf4 ) .                               | 0.766              | 0.554             | 
| NFKB2        | RELB         | the results indicate that dystrophic muscle is characterized by increases in the whole cell expression of ikappab-alpha , p65 , p50 , [relb]{.gene_color} , [p100]{.gene_color} , p52 , ikk , and traf-3 .                                                         | 0.76               | 0.553             | 
| SSSCA1       | CDKN1B       | conclusion : hl-60 / ht cells have lower [p27 (]{.gene_color} [kip1 )]{.gene_color} expression compared with hl-60 cells .                                                                                                                                         | 0.757              | 0.552             | 
| PTH2R        | PTH2         | thus , the juxtamembrane receptor domain specifies the signaling and binding selectivity of [tip39]{.gene_color} for the [pth2 receptor]{.gene_color} over the pth1 receptor .                                                                                     | 0.749              | 0.55              | 
| MMP9         | MMP2         | all these factors markedly influenced the secretion and/or activation of [mmp-2]{.gene_color} and [mmp-9]{.gene_color} .                                                                                                                                           | 0.738              | 0.547             | 
| CCND1        | ABL1         | synergy with [v-abl]{.gene_color} depended on a motif in [cyclin d1]{.gene_color} that mediates its binding to the retinoblastoma protein , suggesting that abl oncogenes in part mediate their mitogenic effects via a retinoblastoma protein-dependent pathway . | 0.736              | 0.547             |
Table: Contains the top ten Gene-interacts-Gene confidence scores before and after model calbration. Both gene mentions highlighted in [blue]{.gene_color}.{#tbl:gg_top_ten_table}

| Gene1 Symbol | Gene2 Symbol | Text                                                                                                                                                                                                                                                       | Before Calibration | After Calibration | 
|--------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------| 
| IFNG         | IL6          | in the control group , the positive rate for il-4 , [il-6]{.gene_color} , il-10 were 0/10 , 2/10 and 1/10 , respectively , and those for il-2 and [ifn-gamma]{.gene_color} were both 1/10 .                                                                      | 0.012              | 0.306             | 
| ACHE         | BCHE         | anticholinesterase activity was determined against acetylcholinesterase ( [ache )]{.gene_color} and butyrylcholinesterase ( [bche )]{.gene_color} , the enzymes vital for alzheimer 's disease , at 50 , 100 and 200 g ml ( -1 ) .                               | 0.011              | 0.306             | 
| CCL2         | AGT          | we found no significant increase in [mcp-1]{.gene_color} concentrations by [ang ii]{.gene_color} alone ; but it enhanced the tnf-alpha-induced mcp-1 mrna expression in a dose-dependent manner .                                                                | 0.011              | 0.306             | 
| CXCL8        | IL1B         | furthermore , somatostatin completely abrogated the increased secretion of [il-8]{.gene_color} and [il-1beta]{.gene_color} after invasion by salmonella .                                                                                                        | 0.011              | 0.303             | 
| SULT1A2      | SULT1A3      | to date , the laboratory has cloned seven unique human sulfotransferases ; five aryl sulfotransferases ( hast1 , hast2 , [hast3]{.gene_color} , [hast4]{.gene_color} and hast4v ) , an estrogen sulfotransferase and a dehydroepiandrosterone sulfotransferase . | 0.009              | 0.295             | 
| IFNG         | IL10         | results : we found weak mrna expression of interleukin-4 ( il-4 ) and il-5 , and strong expression of il-6 , [il-10]{.gene_color} and [ifn-gamma]{.gene_color} before therapy .                                                                                  | 0.008              | 0.292             | 
| IL2          | IFNG         | prostaglandin e2 at priming of naive cd4 + t cells inhibits acquisition of ability to produce [ifn-gamma]{.gene_color} and [il-2]{.gene_color} , but not il-4 and il-5 .                                                                                         | 0.007              | 0.289             | 
| IL2          | IFNG         | the detailed distribution of lymphokine-producing cells showed that [il-2]{.gene_color} and [ifn-gamma-producing]{.gene_color} cells were located mainly in the follicular areas .                                                                               | 0.007              | 0.287             | 
| IL2          | IFNG         | pbl of ms patients produced more pro-inflammatory cytokines , [il-2]{.gene_color} , [ifn-gamma]{.gene_color} and tnf/lymphotoxin , and less anti-inflammatory cytokine , tgf-beta , during wk 2 to 4 in culture than pbl of normal controls .                    | 0.006              | 0.283             | 
| NFKB1        | TNF          | [nf-kappab-dependent]{.gene_color} reporter gene transcription activated by [tnf]{.gene_color} was also suppressed by calagualine .                                                                                                                              | 0.005              | 0.276             | 
Table: Contains the bottom ten Gene-interacts-Gene confidence scores before and after model calbration. Both gene mentions highlighted in [blue]{.gene_color}.{#tbl:gg_bottom_ten_table}


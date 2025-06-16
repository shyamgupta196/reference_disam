
# Claim Source Retrieval


## Description

Given an implicit reference to a scientific paper, i.e., a social media post (tweet) that mentions a research publication without a URL, this method enables to retrieve the mentioned paper from a pool of candidate papers. It was initially developed to leverage [CORD19](https://github.com/allenai/cord19), a corpus of academic papers about COVID-19 and related coronavirus research, however, it can be used with any corpus of publications with enough metadata. 

The method takes an input claim or sentence from the user, computes its similarity with the publication titles and abstracts in the corpus, and returns a ranked list of matching publications. The similarity between the input claim and the publications is calculated using  [BM25](https://en.wikipedia.org/wiki/Okapi_BM25).


## Keywords

source retrieval, scientific references, information retrieval

## Social Science Use Case(s)
1. A social scientist wants to find which publication is possible mentioned in a claim/statement. 
2. A social scientist wants to find topically similar publications to a claim/statement.



## Repository Structure

[](https://github.com/YSKartal/reference_disam#structure)

The repository is organized as follows:

-   Files
	-  [notebooks/getting_started_claim_source.ipynb](https://github.com/YSKartal/reference_disam/blob/main/notebooks/getting_started_claim_source.ipynb): The main notebook file to run the project
    -   [collection_data.py](https://github.com/YSKartal/reference_disam/blob/main/data/collection_data.pkl): The sample publication corpus
    -   [query_posts.tsv](https://github.com/YSKartal/reference_disam/blob/main/data/query_posts.tsv): Dataset of sample posts/claims


## Environment Setup

[](https://github.com/YSKartal/reference_disam#environment-setup)

-   This method can be run with [Jupyter](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/) requiring min Python 3.
 
    

## How to Use

[](https://github.com/YSKartal/reference_disam#how-to-use)

Please follow the instructions in the notebook
    

## Input Data


### Sample Input

     Published in the journal Antiviral Research, the study from Monash University showed that a single dose of Ivermectin could stop the coronavirus growing in cell culture -- effectively eradicating all genetic material of the virus within two days. 
---
    Peer-reviewed in the New England Journal of Medicine regarding Delta (B.1.617.2):  
    •Pfizer is ~90% effective  
    •AstraZeneca is ~70% effective.  
    This falls in line with vaccine efficacy of other variants. Yes, the vaccines ARE indeed effective against Delta.

## Output Data


### Sample Output

This output aims to show an example publication matching for the given input. The real output of the method is seen in different format 

    Published in the journal Antiviral Research, the study from Monash University showed that a single dose of Ivermectin could stop the coronavirus growing in cell culture -- effectively eradicating all genetic material of the virus within two days.: 5g02ykhi -   
    Effectiveness of Covid-19 Vaccines against the B.1.617.2 (Delta) Variant

---

    Peer-reviewed in the New England Journal of Medicine regarding Delta (B.1.617.2):  
        •Pfizer is ~90% effective  
        •AstraZeneca is ~70% effective.  
        This falls in line with vaccine efficacy of other variants. Yes, the vaccines ARE indeed effective against Delta.: ivy95jpw - The FDA-approved drug ivermectin inhibits the replication of SARS-CoV-2 in vitro
    
## Limitation

[](https://github.com/YSKartal/reference_disam#limitation)

## Contact

For questions or feedback, contact Yavuz Selim Kartal via [YavuzSelim.Kartal@gesis.org](mailto: YavuzSelim.Kartal@gesis.org).

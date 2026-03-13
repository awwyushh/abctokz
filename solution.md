# Augenblick Tasks

**Author:** Trained Models  
**Date:** March 2026  

---

## Introduction

This is the solution document for the `Tasks.md` file.

---

## Understanding Behind Tokenization

Machine learning models and Artificial Intelligence, sophisticated as they might be, are just maths in disguise as per our group's philosophy, or the general truth rather. What we wish to accomplish by stating this is that the input–output mappings are ultimately **real numbers**, not textual context.

We define a new term here for further learnings.

**Embeddings:**  
A representation of a word.

Okay, simple. So what?

We understand that words can be assigned some number in context by providing a mapping of each word to a specific number or real value.

However, the commonly used definition in practice is slightly different: a representation is typically a **real-valued vector that encodes the meaning of a word** such that words closer in the vector space are expected to be similar in meaning.

Now, generation of these mappings is not that easy.

Yes — it is mathematics again.

Common techniques include:

- Neural networks
- Dimensionality reduction on word co-occurrence matrices
- Probabilistic models
- Explainable knowledge-base methods
- Explicit representations based on the contexts in which words appear

Reference:  
https://en.wikipedia.org/wiki/Word_embedding

---

## Understanding of Codebase

### Environment Setup

We observed the presence of a `pyproject.toml`, so it was straightforward to set up the environment using:

we also observed in the src directory there is normalisation,model,processing,pre-processing,decoding, eval and training folders, These folders basically highlight the overview the basic, flow of the input text in the tokenisation workflow

we also observe an adapters folder, which comprises of other external tokenisers. to which we can run benchmarks and compare results.

before we can move on to the task lets run this, There are two ways to run this, by the command line and but running a file

## 1. Using the example files to encode,decode and benchmark:
- Ran the example files i.e the train_bpe,train_unigram and train_wordlevel file by doing uv run <file_name>
SO the understanding is that when input as text is received, it is first **normalised --> pre-processed --> tokenisation model(bpe|wordlevel|unigram) --> post-processinf --> output**

Also the benchmarks show metrics like **Tokenizer | Lang | Sentences | Throughput (sps) | Mean tokens/sent | Fertility | UNK rate | Round-trip% | Seq-len ratio |** etc

Now that we have a basic understanding of how this codebase works and runs we can move on to work on the Tasks.

# Task 1[Trace how the bpe tokenisations is working from enocode() to decode()]:
So to understand this we wrote a small script altering the predefined example file train_bpe.py, 
The corpus has mixed Hindi English and Marathi Language
Okay so upon running this we get a series of almost 76 tokens
<img width="1014" height="191" alt="image" src="https://github.com/user-attachments/assets/c5a697a9-c322-4ca8-b519-773ff1b8884a" />

so my understanding is that from when the encode is run we first land at tokeniser.py file(class AugenblickTokenizer) here the sequential pipeline is run **Normalization -> Pre-tokenization -> Modeling -> Post-processing** 
here after teh normalisation includes the unicode normalisation file i.e NFKC text
for such fully Devangari sequence it will basically remain unchanged
in the pretokenisation, coming out from the whitespavepretokenizer it will be split at places of newline and whitespace characters. 
next would be the BPE model. So this is basically a subword splitting tokeniser, its gonna break down incoming text to near individual tokens and then start grouping subwords according to which subwords are repeated, It gives the tokens that reapeat frequently a new token recursively while its exhausted.
then the final step in the encoding, it looks up the vocabulary table to , turn each and every subwords to individual ids.

The decoding back to string is dont by the subwordDecoder, it taked the integer ids and converts them back to the subwords from the vocabulary table, and the special charecters are substituted. 

An observation here was that there were a bunch of 0s inside signifying unknown tokens and there was a stark difference in the original text and the final decoded text.


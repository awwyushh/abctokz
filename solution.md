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

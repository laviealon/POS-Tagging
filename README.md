# Stochastic POS Tagger / Computational Linguistics Project

In my 3rd year of my BSc Mathematics & Computer Science at the University of Toronto, we build a part-of-speech (PoS) tagger.

A PoS tagger classifies the roles tokens in a text are playing, such as "NN" (noun) and "VBD" (verb, past tense). Before tansformer models, this was fundamental to NLP, but continues to have various usages such as model interprability and data enrichment. The main challenge is handling ambiguity.  Consider the word "might": our PoS tagger needs to be able to identify whether to tag it as "NN" or as "VBD". This is achieved via a stochastic hidden Markov model (HMM), which is trained on pre-tagged text corpa to determine context-dependent tag probabilities. 

The tagger (`tagger.py`) is trained on the two training corpa and can be tested on any test corpus. It achieved >94% accuracy when tested in the course.

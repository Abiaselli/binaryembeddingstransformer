# Binary Level Embeddings Transformer Trainer
Binary embeddings transformer trainer

This is a tokenizer free training and inference program I am working on that uses 32 bit segments of binary instead of tokens. Basically, it is set up to seek a value in the logits where you set a threshold, above which it sets that bit at each position as a 1, and it takes 8 bit sections and turns those into letters making each 32 bit segment 4 letters. 

Currently, I am having an issue where the loss seems to be leading the values either strictly towards positive infinity or negative infinity. It should output logits around 0-1 (really these are the pre-sigmoid values so between -5 and 5 or something in that range, then the sigmoid is performed and the result is taken) but it has been issuing basically all 0's. This may be an issue with the loss function. BCE with logits is supposed to do the sigmoid first but maybe it is doing it at the wrong spot or dimension, there might be a squeezing issue. I will try doing the sigmoid first then using regular bce and seeing if I have better luck. 

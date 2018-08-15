# Debugging constant output caption

Currently the trained model always outputs the same caption, independently of
input image. Why is this the case and how to solve it?


## Zero/constant output from imagenet

Hypothesis: the output from the intial imagenet layer is constant / zero and
not taken into account.

Why: because the model is just learning natural language, rather than the
actual content of the images. 

Possible solution: don't update the weights of the imagenet layer. Only 'learn'
how to decode the class descriptions. 

**Answer**: yes this is indeed the case! The image layer always gives a
constant output, independently of input image. 

## Code: what is the target?

input: embedding | input-text
target: input-text

e.g. emb | a man is sitting
     a     man is sitting

OK.

## Optimizing for 'a'

Hypothesis: the weights for the output of the imagenet layer are being
over-optimised for the most likely first word, 'a'. The fact that there is a
loss recorded at this point is counterproductive and shouldn't be the case.

To test: 681 is the argmax of the image layer. Through the LSTM and embedding,
softmax, this gets converted to something fixed. Is it the 'a'?

**Answer**: yes! Because the sentence that gets generated always starts with
'a': 'a man is standing on a skateboard in a park'.

Possible solution: eliminate the loss associated with the output to the
imagenet layer. Opt for the 'start word' solution, as in the diagram of the
paper. But why is this better than directly using the embedding as a first
word? Intuitively, we want the imagenet weights to be affected less. Would this
be the case with the start word mechanism?

Handwavy ideas about this. Key idea is that a learning algorithm splits into
different parts, and one of them, in this particular case, is to inject the
output of the imagenet layer into the state of the LSTM. This injection
requires some transation and so we must be careful and choose the right
learning pipeline. Apparently using the output of the imagenet model as input
to the LSTM provides the wrong incentive and is the wrong way to inject prior
information into the state. 

Either way: the hypothesis is that mocking the embedding as input is the wrong
way to inject prior information into the state and that instead the 'start
word' mechanism must be used instead. 

So: implement the 'input word' strategy and see if this eliminates the
'mapping everything to a' problem. 

Implementation using keras: use a start word by extending the vocabulary size
by 1, and without changing the training data. The output of the RNN is still a
single tensor with the number of time steps in the second dimension. Simply
apply a custom layer that drops the first time step. 

**This update solves the constant output problem!** However, unfortunately most
of the captions are still inaccurate and the predictions are poor. Still only a
handful of sentences appear, and these *sometimes* happen to be accurate, but
not sufficiently so -- there is only 1 example I've found to be convincing.
The results seem a little bit coarse, as though more training would do the
trick.

## Increasing the training time.

Hypothesis: increased training will improve the results.

This is suggested by the fact that there's a handful of sentences that appear
frequently, and so the results look quite 'coarse', as though only broad
clusters have been identified. Increased training suggests we might be able to
improve results. 

As there is more training, we still get certain sentences appearing very
frequently, except that now it's different sentences. Is the randomization
bracket too small? Training data should be shuffled, so this shouldn't be a
problem. 


## Overfitting

The paper quotes overfitting. Do I get much better results by running on actual
training images? What does the tensorboard graph look like once training is
complete?


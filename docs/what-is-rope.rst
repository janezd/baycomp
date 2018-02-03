Understanding the meaning of `rope`
===================================

When comparing two classifiers on a single data set, two classifiers are
considered equivalent if the difference in their performance, according to
some score, is below some threshold. For instance, if a difference in
classification accuracies is below 0.5 percent, it may be considered
negligible.

For comparison on multiple data sets, the interpretation of rope is a bit
different.

Here, we are considering the probabilities that, given a new data set
(of a similar kind as data sets on which we compared the classifiers so far),
one learning algorithm will perform better than another. So, say that for some
learning algorithms A and B, we know their performance on 50 data sets.
Algorithm A may had been better on 30 data sets, so the probability that
it is indeed better than B, may be 0.6. However, with these experiments
we can use Monte Carlo sampling to construct a posterior with, say 1000
samples, which correspond to 1000 (hypothetical) new experiments on 50 data
sets (which we have not actually done). In every "experiment", we get a
different probability that A is better than B; most are close to 0.6, but
sometimes it may be as low as 0.51 or, in some cases, B can be even better,
so the probability for A may be just, say, 0.4.

The way in which these 1000 samples were obtained is not important here;
this is what the three methods (the sign test, the sign rank test and
the hierarchical t test) differ in. We also won't discuss priors here.

Now comes the rope: we will say that if the probability of A being better is
below 0.55, we will not claim it is actually better. Similarly, we won't
proclaim B the winner in an experiment unless it grabs at least 0.55 probability.
Hence, we have a rope of 0.05.

Say that in 700 experiments (out of 1000), the probability for A being better
was above 0.55. For these cases we proclaim A the winner. The probability that
A is better is thus 70 %.

Further, say that in 200 experiments neither classifier achieved 0.55
probability of being better than the other. Hence, there is 20 % chance they
are within the rope, 0.5.

In the remaining 100 cases, B was better; so the probability that B is indeed
better is 0.1.

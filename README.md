[![Stories in Ready](https://badge.waffle.io/PFCM/482-project.png?label=ready&title=Ready)](https://waffle.io/PFCM/482-project)

# math482 project

## some kind of monte carlo tree search

Implementation of classic UCT for 9x9 Go and Hex (via [gym](https://github.com/openai/gym)). Single threaded and rather slow.

## other things
Also machinery to train a convolutional neural network to play Hex (again via gym, although involved some slight hacking so 
it is best to use [this fork](https://github.com/pfcm/gym). We can then throw the network into the MCTS, specifically to
get less noisy rollouts.

## Results

The first major test was whether transposition tables helped UCT play hex. The answer is not particularly, although they
certainly don't hurt. In a tournament of 100 games, UCT with tables beat UCT without 55-45.

Subsequently a couple of tournaments were played between the following algorithms:
 - *random*: chooses a move uniformly at random
 - *uct*: classic UCT (uses UCB-1 in the tree policy)
 - *policy-net-random*: a convolutional network trained with policy gradients against a random opponent (for ≈16000 parameter
    updates, approximately 14000 games).
 - *policy-net-tuned*: the same convolutional network trained for a further 30000 parameter updates (around 20000 games)
    against a running average of its own parameters.
 - *mcts-policy-net-random*: UCT, but with the *policy-net-random* used during rollouts instead of *random*.
 - *mcts-policy-net-tuned*: UCT, but with *policy-net-tuned* used during rollouts instead of *random*.

*random* has no parameters. *uct* and *mcts-&ast;* need parameters `search_time` and `tree_depth` -- the first is literally
the time in seconds it spends searching (and therefore the major limit on actually validating these agents). The second is
how far from the root the tree is allowed to grow. For the classic UCT algorithm, this should be larger than the maximum
depth of the game so that the tree is unconstrained. In practice it doesn't seem to make too much difference, although in
validating the algorithms I've rarely given UCT enough time to saturate and come close to expanding the whole tree even when
testing with quite a small tree depth (the trees are very wide, especially early on). 

The policy nets can be used without parameters. The final architecture is:
````
  [9x9x3] -> [3x3x3x64] -> 3*[3x3x64x64] -> 256                  -> 81 
   input  ->  conv/relu ->    conv/relu  -> fully connected/relu ->  fully connected
````
They were trained with RMSProp with quite a low learning rate and a very small amount of weight decay (l_2 regularisation).

####Tournaments

Three round-robin tournaments were run on a few different machines (ECS and my personal computer) with a couple of
different settings. For the round-robins, 7 games were played and the algorithm that won the majority was awarded
one point. Settings changed between runs (apart from switching computers at one point) were the MCTS search time and maximum
depth and these were applied globally to all three variants for the duration of the tournament.

#####Round 1 - ECS machines (10 second search, unlimited tree depth)
(rollouts for the neural-net mcts algorithms took up to 0.09 seconds each, while standard UCT was more like 0.004)

|Player 1 | Player 2 | Score |
|---------|----------|-------|
|*mcts-policy-net-random* | *mcts-policy-net-tuned* | 2-5 |
|*random* | *policy-net-random* | 4-3 |
|*random* | *policy-net-tuned* | 0-7 |
|*random* | *uct* | 2-5 |
|*random* | *mcts-policy-net-random* | 4-3 |
|*random* | *mcts-policy-net-tuned* | 4-3 |
|*uct*    | *mcts-policy-net-random* | 4-3 |
|*uct*    | *mcts-policy-net-tuned* | 2-5 |
|*uct*    | *policy-net-random* | 5-2 |
|*uct*    | *policy-net-tuned* | 4-3 |
|*policy-net-random* | *mcts-policy-net-random* | 6-1 |
|*policy-net-random* | *mcts-policy-net-tuned* | 5-2 |
|*policy-net-random* | *policy-net-tuned* | 1-6 |
|*policy-net-tuned* | *mcts-policy-net-random* | 4-3 |
|*policy-net-tuned* | *mcts-policy-net-tuned* | 7-0 |

|Algorithm | Wins |
|----------|------|
|*random* | 3 |
|*uct* | 4 |
|*policy-net-random* | 2 |
|*policy-net-tuned* | 4 |
|*mcts-policy-net-random* | 0 |
|*mcts-policy-net-tuned* | 2 |

##### Round 2 - Home computer (NVIDIA GeForce GTX 970, otherwise essentially the same) (10 sec search, max tree depth 10)
(using a discrete graphics card for the convnets cut the time down to around 0.03 seconds per rollout, still an order of
magnitude slower than the uniform rollouts)

|Player 1 | Player 2 | Score |
|---------|----------|-------|
|*mcts-policy-net-random* | *mcts-policy-net-tuned* | 3-4 |
|*random* | *policy-net-random* | 4-3 |
|*random* | *policy-net-tuned* | 0-7 |
|*random* | *uct* | 2-5 |
|*random* | *mcts-policy-net-random* | 5-2 |
|*random* | *mcts-policy-net-tuned* | 2-5 |
|*uct*    | *mcts-policy-net-random* | 4-3 |
|*uct*    | *mcts-policy-net-tuned* | 6-1 |
|*uct*    | *policy-net-random* | 4-3 |
|*uct*    | *policy-net-tuned* | 2-5 |
|*policy-net-random* | *mcts-policy-net-random* | 2-5 |
|*policy-net-random* | *mcts-policy-net-tuned* | 3-4 |
|*policy-net-random* | *policy-net-tuned* | 1-6 |
|*policy-net-tuned* | *mcts-policy-net-random* | 5-2 |
|*policy-net-tuned* | *mcts-policy-net-tuned* | 4-3 |

|Algorithm | Wins |
|----------|------|
|*random* | 2 |
|*uct* | 4 |
|*policy-net-random* | 0 |
|*policy-net-tuned* | 5 |
|*mcts-policy-net-random* | 1 |
|*mcts-policy-net-tuned* | 3 |

##### Round 3 - Home computer (search time 15 seconds, unlimited tree depth)

|Player 1 | Player 2 | Score |
|---------|----------|-------|
|*mcts-policy-net-random* | *mcts-policy-net-tuned* | 5-2 |
|*random* | *policy-net-random* | 4-3 |
|*random* | *policy-net-tuned* | 1-6 |
|*random* | *uct* | 3-4 |
|*random* | *mcts-policy-net-random* | 2-5 |
|*random* | *mcts-policy-net-tuned* | 2-5 |
|*uct*    | *mcts-policy-net-random* | 4-3 |
|*uct*    | *mcts-policy-net-tuned* | 2-5 |
|*uct*    | *policy-net-random* | 4-3 |
|*uct*    | *policy-net-tuned* | 1-6 |
|*policy-net-random* | *mcts-policy-net-random* | 4-3 |
|*policy-net-random* | *mcts-policy-net-tuned* | 6-1 |
|*policy-net-random* | *policy-net-tuned* | 1-6 |
|*policy-net-tuned* | *mcts-policy-net-random* | 7-0 |
|*policy-net-tuned* | *mcts-policy-net-tuned* | 7-0 |

|Algorithm | Wins |
|----------|------|
|*random* | 1 |
|*uct* | 2 |
|*policy-net-random* | 2 |
|*policy-net-tuned* | 5 |
|*mcts-policy-net-random* | 2 |
|*mcts-policy-net-tuned* | 2 |


### Conclusions
*policy-net-tuned* is clearly the best, it is a little bit disappointing using such a strong policy inside the MCTS
fails to equal the performance of the policy itself. This implies I was in fact incorrect in my
intuition that making the rollout more accurate was a good place to improve the performance of the search. All of the 
results above were achieved with very short search times. Inuitively this should hurt UCT the most, but actually increasing
the search time seemed to make it perform worse. If anything, this most likely implies the search times are still far too
short and that 7 games is not enough to really compare two agents.

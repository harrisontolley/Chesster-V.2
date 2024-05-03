# Outline

The original plan was to create a chess engine that could play against a human player. I wanted to implement this bot onto my own personal portfolio website, and allow users to play against it. Essentially, it's just a project I wanted to do for fun, and to learn more about chess engines, and AI in general, with the added benefit of being able to show it off on my website (and possibly recruiters?).

## Current Version

I have implemented a static board evaluation method, which is used to evaluate the board state at any given time. The evaluation method is based piece material advantage, piece position advantage, and king safety, (isolated/passed) pawn structure.

The move search function is implemented using the minimax algorithm with alpha-beta pruning, transposition tables, and zobrist hashing. The search function is recursive and searches the game tree to a certain depth. The search function also uses the evaluation function to evaluate the board state at the leaf nodes of the game tree. The nature of the minimax algorithm is such that it will always choose the best move for the player, assuming the opponent also plays the best move. In most cases, the opposing player (if human) will not play the best move, and the AI will be able to take advantage of this.

## Future Plans

The limiting factor now is the depth of the search, and the accuracy of the evaluation function. I am in the process of implementing a neural network to replace the evaluation function, using stockfish's evaluation score as the target. This will allow the AI to learn from the stockfish engine, and improve its evaluation function over time.

The issue that I am currently experiencing is that Python is just too slow for training. As well as this, due to the nature of the problem (complex), the network takes a long time to train, and requires copious amounts of data. As I am wanting to do this myself, I am looking into creating a C++/Rust neural network trainer that can be used to train the network on publicly available data. This will allow me to train the network faster, and with more data, and will allow me to implement the network into the AI.

Another bottleneck I am experiencing is that, again, Python is too slow, and takes too long to search the game tree (millions of positions after a few moves). I am looking at alternatives.

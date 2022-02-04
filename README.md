# boards
General framework for playing board games. The interest is in AI for playing games.

## Brief overview of structure
The important classes are as follows
- **bid**: A metaclass to construct a new type representing a boards metastate. When implementation modules are imported they construct the necessary representations of a board state, usually including a string representation, numpy array representation, and 2-char representation for serialization. Since bids are instances of type, we check for moves and board metastates via reference comparison rather than comparing the underlying data, which is never necessary.

- **board**: A class containing relevant data to a single state of a game. Generally has a **bid** associated to it and some object to hold other information. This is the central data structure in the entirety of the program, and essentially all other logic manipulates boards. Boards know how to make moves, get valid moves, and check for terminal board states. They hold the games logic.

- **player**: While **board**s contain game logic, **player**s containg decision making logic. A player must be instantiated with a reference to a relevant board to make their decisions on. A **player** chooses a move on a board (the board actually makes the move, we don't want players doing this for synchronization reasons) and receives information about other players moves. For example, the **mcts** class needs to be updated when another player makes a move so that it can update it's tree. It doesn't want it's tree to be public, so it stays synchronized via the **player.inform** method instead.

These are by far the most important classes to structure. There are also classes still finding their purpose, such as **elo**, **game**, and **model**, which manage logic of those subjects. Once the training algorithms solidify more the logic for presenting and playing games to end users will be developed more, and tracking a specific AIs **elo** will become more interesting. For now the focus is on developing the relevant training structure and training the models.

## Current state of development
As of 2/4/2022, the project is successfully training an MCTS keras model on connect 4. The immediate goals are to continue training and begin comparing the model to other players such as human players and minimax players for benchmarking. Longer term goals include optimizing training, improving model architecture, and creating a UI.
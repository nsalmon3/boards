# boards
General framework for playing board games. The interest is in AI for playing games.

## General structure
The general structure of this code is based on players and boards. Players contain logic to pick a move given a board.
A board generally consists of some object to store the state, and a bid object to store the metastate.

For example, in connect 4 the state is a (6, 7) array, and the bid is one of 7 metastates, e.g. RED_TO_MOVE
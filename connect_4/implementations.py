from importlib_metadata import itertools
import numpy as np

from abstract import *
from exceptions import *

# Non-terminal board ids
BLACK_TO_MOVE = 'B'
RED_TO_MOVE = 'R'
BLACK_OFFERS_DRAW = 'BD'
RED_OFFERS_DRAW = 'RD'

# Terminal board ids
BLACK_WINS = 'BW'
RED_WINS = 'RW'
DRAW = 'D'

# String representations of board ids
ID_STRINGS = {
    BLACK_TO_MOVE: "Black to move.",
    RED_TO_MOVE: "Red to move.",
    BLACK_OFFERS_DRAW: "Black offers draw.",
    RED_OFFERS_DRAW: "Red offers draw.",

    BLACK_WINS: "Black wins.",
    RED_WINS: "Red wins.",
    DRAW: "Draw."
}

# Short code representation of board ids (mostly for serialization)
ID_CODES = {
    BLACK_TO_MOVE: "B",
    RED_TO_MOVE: "R",
    BLACK_OFFERS_DRAW: "BD",
    RED_OFFERS_DRAW: "RD",

    BLACK_WINS: "BW",
    RED_WINS: "RW",
    DRAW: "D"
}

# Should be the opposite dictionary of above (mostly for fast deserialization)
ID_UNCODES = {
    "B": BLACK_TO_MOVE,
    "R": RED_TO_MOVE,
    "BD": BLACK_OFFERS_DRAW,
    "RD": RED_OFFERS_DRAW,

    "BW": BLACK_WINS,
    "RW": RED_WINS,
    "D": DRAW
}

# Piece-placing move ids
PLACE_1 = "1"
PLACE_2 = "2"
PLACE_3 = "3"
PLACE_4 = "4"
PLACE_5 = "5"
PLACE_6 = "6"
PLACE_7 = "7"

PLACE_INDICES = {
    PLACE_1: 0,
    PLACE_2: 1,
    PLACE_3: 2,
    PLACE_4: 3,
    PLACE_5: 4,
    PLACE_6: 5,
    PLACE_7: 6
} # I feel like this is sort of awkward, so the second I have a better idea I'll probably change it

# Non-piece-placing move ids
OFFER_DRAW = "OD"
ACCEPT_DRAW = "AD"
DECLINE_DRAW = "DD"
RESIGN = "R"

# Used to display moves in a text format
MOVE_STRINGS = {
    PLACE_1: "Place 1",
    PLACE_2: "Place 2",
    PLACE_3: "Place 3",
    PLACE_4: "Place 4",
    PLACE_5: "Place 5",
    PLACE_6: "Place 6",
    PLACE_7: "Place 7",
    OFFER_DRAW: "Offer draw",
    ACCEPT_DRAW: "Accept draw",
    DECLINE_DRAW: "Decline draw",
    RESIGN: "Resign"
}

# For fast uncoding of a move string (don't actually think speed matters but it's nice)
MOVE_UNSTRINGS = {
    "Place 1": PLACE_1,
    "Place 2": PLACE_2,
    "Place 3": PLACE_3,
    "Place 4": PLACE_4,
    "Place 5": PLACE_5,
    "Place 6": PLACE_6,
    "Place 7": PLACE_7,
    "Offer draw": OFFER_DRAW,
    "Accept draw": ACCEPT_DRAW,
    "Decline draw": DECLINE_DRAW,
    "Resign": RESIGN
}

# Fun display colors for command line
COLOR = {
    "R": "\033[91m",
    "B": "\033[94m",
    "end": "\033[0m"
}

class connect_4_board(board):
    """
    This is the class containing all relevant logic to a Connect 4 board.
    """

    def __init__(self, grid: np.array = None, id: str = None, *args, **kwargs):
        if grid is None:
            grid = np.full((6, 7), 0)
        if id is None:
            id = BLACK_TO_MOVE

        super().__init__(*args, grid = grid, id = id, **kwargs)
    
    def __str__(self):
        _bs = "_________________\n"
        for i in range(6):
            _bs += "| "
            for j in range(7):
                if self[i,j] == -1:
                    _bs += COLOR["R"] + 'R ' + COLOR["end"]
                elif self[i,j] == 1:
                    _bs += COLOR["B"] + 'B ' + COLOR["end"]
                else:
                    _bs += 'O '
            _bs += "|\n"
        _bs += "\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e"
        return (
            ID_STRINGS[self.id] + '\n' +
            _bs + '\n' +
            "Available moves: " + str([MOVE_STRINGS[move] for move in self.moves])
        )
            
    def __getitem__(self, item):
        return self.grid[item]
    
    def __setitem__(self, item, val):
        self.grid[item] = val

    @property
    def is_terminal(self) -> bool:
        return (
            self.id == BLACK_WINS or
            self.id == RED_WINS or
            self.id == DRAW)
    
    @property
    def moves(self) -> 'list[str]':
        l = list()

        # First check to see if the board is in a terminal state.
        if self.is_terminal: return l

        # Then see if a draw was offered.
        if self.id == BLACK_OFFERS_DRAW or self.id == RED_OFFERS_DRAW: 
            return [ACCEPT_DRAW, DECLINE_DRAW]

        # Next we loop through the top of the board to see which columns have openings.
        for i in range(len(self[0, :])):
            if self[0, i] == 0:
                l.append(eval(f"PLACE_{i + 1}")) # This safely allows for us to change the id encoding later

        # Finally add draw offer and resign
        l.append(OFFER_DRAW)
        l.append(RESIGN)

        return l

    def copy(self) -> 'connect_4_board':
        return type(self)(grid = np.copy(self.grid), id = self.id)
    
    def reset(self) -> None:
        self.grid = np.full((6, 7), 0)
        self.id = BLACK_TO_MOVE

    def move(self, _move: str, inplace: bool = True) -> 'connect_4_board':
        # If the move is not one of the valid moves then throw an exception.
        # Note that this catches moves attempted in terminal board states.
        if _move not in self.moves:
            raise InvalidMoveException(_move + " is an invalid move in the board state: " + ID_STRINGS[self.id])

        # If the operation is not in place, then copy the board to return a new copy
        if inplace: _board = self
        else: _board = self.copy()

        # First check non piece-placing moves
        # Since the first line validates the move for the board state, we can safely assume the correct logic below
        if _move == OFFER_DRAW:
            if _board.id == BLACK_TO_MOVE:
                _board.id = BLACK_OFFERS_DRAW
            else:
                _board.id = RED_OFFERS_DRAW
            return _board
        elif _move == ACCEPT_DRAW:
            _board.id = DRAW
            return _board
        elif _move == DECLINE_DRAW:
            if _board.id == BLACK_OFFERS_DRAW:
                _board.id = BLACK_TO_MOVE
            else:
                _board.id = RED_TO_MOVE
        elif _move == RESIGN:
            if _board.id == RED_TO_MOVE or _board.id == BLACK_OFFERS_DRAW:
                _board.id = BLACK_WINS
            else:
                _board.id = RED_WINS
            return _board

        # Loop through the column and set the last open spot with the color
        # At this point we know we have PLACE_n as the move and
        # BLACK_TO_MOVE or RED_TO_MOVE as the state
        j = PLACE_INDICES[_move]
        if _board.id == BLACK_TO_MOVE:
            c = 1
        else:
            c = -1
        for i, spot in enumerate(_board[:, j]):
            if spot != 0:
                _board[i - 1, j] = c
                break
            elif i == 5:
                _board[i, j] = c

        # Now we must figure out the id of this new board state
        # First check for horizontal wins
        for i in range(6):
            for j in range(4):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i,j+1] == _board[i,j+2] == _board[i,j+3]:
                    if _board[i,j] == -1:
                        _board.id = RED_WINS
                    else:
                        _board.id = BLACK_WINS
                    return _board
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j] == _board[i+2,j] == _board[i+3,j]:
                    if _board[i,j] == -1:
                        _board.id = RED_WINS
                    else:
                        _board.id = BLACK_WINS
                    return _board
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j+1] == _board[i+2,j+2] == _board[i+3,j+3]:
                    if _board[i,j] == -1:
                        _board.id = RED_WINS
                    else:
                        _board.id = BLACK_WINS
                    return _board

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i-1,j+1] == _board[i-2,j+2] == _board[i-3,j+3]:
                    if _board[i,j] == -1:
                        _board.id = RED_WINS
                    else:
                        _board.id = BLACK_WINS
                    return _board

        # We have no connect 4's. Now check for a draw (no valid moves).
        if _board.moves == []:
            _board.id = DRAW
            return _board
        
        # At this point, a valid piece placing move was made and the board is not in a terminal state.
        # So it's just the other players turn now.
        if _board.id == BLACK_TO_MOVE: _board.id = RED_TO_MOVE
        else: _board.id = BLACK_TO_MOVE

        return _board
    
    def serialize(self) -> str:
        ser = ID_CODES[self.id] + ':'
        for i in range(6):
            for j in range(7):
                ser += str(self[i,j])
        return ser

    @classmethod
    def deserialize(cls, ser: str) -> 'connect_4_board':
        l = ser.split(":")

        _id = ID_UNCODES[l[0]]
        
        it = iter(itertools.product(range(6), range(7)))
        _grid_str = l[1]
        _grid = np.full((6,7), 0)
        i = 0
        while i < len(_grid_str):
            tpl = next(it)
            if _grid_str[i] == "-":
                _grid[tpl[0],tpl[1]] = -1
                i += 2
            else:
                _grid[tpl[0], tpl[1]] = _grid_str[i]
                i += 1
        
        return cls(grid = _grid, id = _id)

class random_player(player):
    """
    This player makes random moves that are not draw offers or resignations
    """
    def move(self) -> str:
        _moves = self.board.moves
        if OFFER_DRAW in _moves: _moves.remove(OFFER_DRAW)
        if RESIGN in _moves: _moves.remove(RESIGN)

        return np.random.choice(_moves)
    
    def inform(self, _move: str):
        ...

class cli_player(player):
    """
    This player plays from the command line interface, inputting moves as strings. 
    """
    def move(self) -> str:
        print(self.board)
        valid = False
        while not valid:
            _move = MOVE_UNSTRINGS[input("Please input a valid move: ")]
            
            if _move not in self.board.moves:
                print("That was not a valid move.")
            else:
                return _move

    def inform(self, _move: str):
        print(MOVE_STRINGS[_move] + " was played.")
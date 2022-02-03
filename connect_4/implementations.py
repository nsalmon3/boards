from importlib_metadata import itertools
import numpy as np

from abstract import *
from exceptions import *

class BLACK_TO_MOVE(metaclass = bid):
    as_array = np.array((1., 0., 0., 0., 0.))
    as_string = "black to move"
    index = 0
    serialized = "BM"

class RED_TO_MOVE(metaclass = bid):
    as_array = np.array((0., 1., 0., 0., 0.))      
    as_string = "red to move"
    index = 1     
    serialized = "RM"

class BLACK_WINS(metaclass = bid):
    as_array = np.array((0., 0., 1., 0., 0.))
    as_string = "black wins"
    index = 2
    serialized = "BW"

class RED_WINS(metaclass = bid):
    as_array = np.array((0., 0., 0., 0., 1.))
    as_string = "red wins"
    index = 3
    serialized = "RW"

class DRAW(metaclass = bid):
    as_array = np.array((0., 0., 0., 0., 1.))
    as_string = "draw"
    index = 4
    serialized = "DR"

class PLACE_1(metaclass = bid): 
    as_array = np.array((1., 0., 0., 0., 0., 0., 0.))
    as_string = "place 1"
    index = 0
    serialized = "P1"

class PLACE_2(metaclass = bid): 
    as_array = np.array((0., 1., 0., 0., 0., 0., 0.))
    as_string = "place 2"
    index = 1
    serialized = "P2"

class PLACE_3(metaclass = bid):
    as_array = np.array((0., 0., 1., 0., 0., 0., 0.))
    as_string = "place 3"
    index = 2
    serialized = "P3"

class PLACE_4(metaclass = bid):
    as_array = np.array((0., 0., 0., 1., 0., 0., 0.))
    as_string = "place 4"
    index = 3
    serialized = "P4"

class PLACE_5(metaclass = bid):
    as_array = np.array((0., 0., 0., 0., 1., 0., 0.))
    as_string = "place 5"
    index = 4
    serialized = "P5"

class PLACE_6(metaclass = bid):
    as_array = np.array((0., 0., 0., 0., 0., 1., 0.))
    as_string = "place 6"
    index = 5
    serialized = "P6"

class PLACE_7(metaclass = bid):
    as_array = np.array((0., 0., 0., 0., 0., 0., 1.))
    as_string = "place 7"
    index = 6
    serialized = "P7"

def deserialize(_id: str):
    # Board ids
    if _id == BLACK_TO_MOVE.serialized: return BLACK_TO_MOVE
    elif _id == RED_TO_MOVE.serialized: return RED_TO_MOVE
    
    elif _id == BLACK_WINS.serialized: return BLACK_WINS
    elif _id == RED_WINS.serialized: return RED_WINS
    elif _id == DRAW.serialized: return DRAW

    # Move ids
    elif _id == PLACE_1.serialized: return PLACE_1
    elif _id == PLACE_2.serialized: return PLACE_2
    elif _id == PLACE_3.serialized: return PLACE_3
    elif _id == PLACE_4.serialized: return PLACE_4
    elif _id == PLACE_5.serialized: return PLACE_5
    elif _id == PLACE_6.serialized: return PLACE_6
    elif _id == PLACE_7.serialized: return PLACE_7

    else: raise ValueError(str(_id) + " could not be deserialized for connect_4.")


# Fun display colors for command line
_COLOR = {
    "R": "\033[91m",
    "B": "\033[94m",
    "end": "\033[0m"
}

class connect_4_board(board):
    """
    This is the class containing all relevant logic to a Connect 4 board.
    """

    def __init__(self, grid: np.array = None, _id: bid = None, *args, **kwargs):
        if grid is None:
            grid = np.full((6, 7), 0.)
        if _id is None:
            _id = BLACK_TO_MOVE

        # The draw flag stops a player from offering a draw after a decline of a draw offer.
        self._draw_flag = False

        super().__init__(*args, grid = grid, bid = _id, **kwargs)
    
    def __str__(self):
        _bs = "_________________\n"
        for i in range(6):
            _bs += "| "
            for j in range(7):
                if self[i,j] == -1:
                    _bs += _COLOR["R"] + 'R ' + _COLOR["end"]
                elif self[i,j] == 1:
                    _bs += _COLOR["B"] + 'B ' + _COLOR["end"]
                else:
                    _bs += 'O '
            _bs += "|\n"
        _bs += "\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e\u203e"
        return (
            self.bid.as_string + '\n' +
            _bs + '\n' +
            "Available moves: " + str([move.as_string + " (" + move.serialized + ")" for move in self.moves]))
            
    def __getitem__(self, item):
        return self.grid[item]
    
    def __setitem__(self, item, val):
        self.grid[item] = val

    @property
    def is_terminal(self) -> bool:
        return (
            self.bid == BLACK_WINS or
            self.bid == RED_WINS or
            self.bid == DRAW)
    
    @property
    def moves(self) -> 'list[object]':
        l = list()

        # First check to see if the board is in a terminal state.
        if self.is_terminal: return l

        # We loop through the top of the board to see which columns have openings.
        for i in range(len(self[0, :])):
            if self[0, i] == 0:
                l.append(eval(f"PLACE_{i + 1}")) # This safely allows for us to change the id encoding later

        return l

    def copy(self) -> 'connect_4_board':
        return type(self)(grid = np.copy(self.grid), _id = self.bid)
    
    def reset(self) -> None:
        self.grid = np.full((6, 7), 0.)
        self.bid = BLACK_TO_MOVE

    def move(self, _move, inplace: bool = True) -> 'connect_4_board':
        # If the move is not one of the valid moves then throw an exception.
        # Note that this catches moves attempted in terminal board states.
        if _move not in self.moves:
            raise InvalidMoveException(_move.as_string + " is an invalid move in the board state: " + self.bid.as_string)

        # If the operation is not in place, then copy the board to return a new copy
        if inplace: _board = self
        else: _board = self.copy()

        # Loop through the column and set the last open spot with the color
        # At this point we know we have PLACE_n as the move and
        # BLACK_TO_MOVE or RED_TO_MOVE as the state
        j = _move.index
        if _board.bid == BLACK_TO_MOVE:
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
                        _board.bid = RED_WINS
                    else:
                        _board.bid = BLACK_WINS
                    return _board
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j] == _board[i+2,j] == _board[i+3,j]:
                    if _board[i,j] == -1:
                        _board.bid = RED_WINS
                    else:
                        _board.bid = BLACK_WINS
                    return _board
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j+1] == _board[i+2,j+2] == _board[i+3,j+3]:
                    if _board[i,j] == -1:
                        _board.bid = RED_WINS
                    else:
                        _board.bid = BLACK_WINS
                    return _board

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if _board[i,j] == 0: continue # Certainly isn't a win
                if _board[i,j] == _board[i-1,j+1] == _board[i-2,j+2] == _board[i-3,j+3]:
                    if _board[i,j] == -1:
                        _board.bid = RED_WINS
                    else:
                        _board.bid = BLACK_WINS
                    return _board

        # We have no connect 4's. Now check for a DRAW (no valid moves).
        if _board.moves == []:
            _board.bid = DRAW
            return _board
        
        # At this point, a valid piece placing move was made and the board is not in a terminal state.
        # So it's just the other players turn now.
        if _board.bid == BLACK_TO_MOVE: _board.bid = RED_TO_MOVE
        else: _board.bid = BLACK_TO_MOVE

        return _board
    
    def jsonify(self) -> dict:
        return {
            'grid': self.grid.tolist(),
            'bid': self.bid.serialized
        }

    @classmethod
    def dejsonify(cls, d: dict) -> 'connect_4_board':
        _grid = np.array(d['grid'])
        _bid = deserialize(d['bid'])
        return connect_4_board(grid = _grid, _id = _bid)

class random_player(player):
    """
    This player makes random moves that are not draw offers or resignations
    """
    def move(self):
        return np.random.choice(self.board.moves)
    
    def inform(self, _move: bid):
        ...

class cli_player(player):
    """
    This player plays from the command line interface, inputting moves as strings. 
    """
    def move(self) -> bid:
        print(self.board)
        valid = False
        while not valid:
            _move = deserialize(input("Please input a valid move: "))
            
            if _move not in self.board.moves:
                print("That was not a valid move.")
            else:
                return _move

    def inform(self, _move: bid):
        print(str(_move) + " was played.")

class connect_4_game(two_player_game):
    def __init__(self, _board: connect_4_board, player_1: player, player_2: player) -> None:
        super().__init__(_board, player_1, player_2)
    
    def play_tourney(self, rounds: int = 1) -> dict:
        draws = 0
        self.player_1.wins = 0
        self.player_2.wins = 0
        for n in range(rounds):
            # Start by initializing the game
            self.board.reset()
            board_list = [self.board.copy()]
            if n % 2 == 0:
                it = iter(cycle([self.player_1, self.player_2]))
            else:
                it = iter(cycle([self.player_2, self.player_1]))
            current_player = next(it)

            while not self.board.is_terminal:
                # Have the player choose a move and make the move
                _move = current_player.move()
                self.board.move(_move)

                # If the move was successful then add the new board state to the list
                board_list.append(self.board)
                
                # Move to the next player
                current_player = next(it)

                # Inform them of the move made by the other player
                current_player.inform(_move)
            if self.board.bid == DRAW:
                draws += 1  
            else:
                current_player.wins += 1
        
        _ = self.player_1.wins - self.player_2.wins
        # Update the elos of the players
        elo.update(self.player_1.elo, self.player_2.elo, _, -_)

        return {
            'player_1': self.player_1.wins,
            'player_2': self.player_2.wins,
            'draws': draws
        }
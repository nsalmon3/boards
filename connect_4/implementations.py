import sys

sys.path.append('..\\boards')

from abstract import *
from exceptions import *
import numpy as np

class connect_4_board(board):
    """
    This is the class containing all relevant logic to a Connect 4 board.
    """
    _ids = {
        # Non-terminal states
        'B': "Black to move.",
        'R': "Red to move.",
        'BD': "Black offers draw.",
        'RD': "Red offers draw.",

        # Terminal states
        'BW': "Black wins.",
        'RW': "Red wins.",
        'D': "Draw."
    }

    def __init__(self, grid: np.array = None, id: str = None, *args, **kwargs):
        if grid is None:
            grid = np.full((6,7), 'O')
        if id is None:
            id = 'B'

        super().__init__(*args, grid = grid, id = id, **kwargs)
    
    def __str__(self):
        return (
            self._ids[self.id] + '\n' +
            str(self.grid) + '\n' +
            "Available moves: " + str(self.moves)
        )
            

    def __getitem__(self, item):
        return self.grid[item]
    
    def __setitem__(self, item, val):
        self.grid[item] = val

    @property
    def is_terminal(self) -> bool:
        return (
            self.id == 'BW' or
            self.id == 'RW' or
            self.id == 'D')
    
    @property
    def moves(self) -> 'list[str]':
        """
        Valid moves for this game are:
        {
            'Rn': Put a red piece in column n,
            'Bn': Put a black piece in column n,
            'OD': Offer a draw,
            'AD': Accept a draw offer,
            'DD': Decline a draw offer,
            'R': Resign
        }
        """
        l = list()

        # First check to see if the board is in a terminal state.
        if self.is_terminal: return l

        # Then see if a draw was offered.
        if self.id == 'BD' or self.id == 'RD': return ['AD', 'DD']

        # We being by adding the draw offer move and resign move.
        l.append('OD')
        l.append('R')

        # Next we loop through the top of the board to see which columns have openings.
        for i in range(len(self[0, :])):
            if self[0,i] == 'O':
                l.append(self.id + str(i + 1)) # self.id is R or B, depending upon whose turn it is.

        return l

    @classmethod
    def copy(cls, _board: 'connect_4_board') -> 'connect_4_board':
        return cls(grid = np.copy(_board.grid), id = _board.id)

    @classmethod
    def move(cls, _board: 'connect_4_board', move: str, inplace: bool = False) -> 'connect_4_board':
        # If the move is not one of the valid moves then throw an exception.
        # Note that this catches moves attempted in terminal board states.
        if move not in _board.moves:
            raise InvalidMoveException(move + " is an invalid move in the current board state!")

        # If the operation is not in place, then copy the board to return a new copy
        if not inplace: _board = cls.copy(_board)

        # First check non piece-placing moves
        if move == 'OD':
            _board.id = _board.id[0] + 'D'
            return _board
        elif move == 'AD':
            _board.id = 'D'
            return _board
        elif move == 'DD':
            _board.id = _board.id[0]
            return _board
        elif move == 'R':
            if _board.id[0] == 'R': _board.id = 'BW'
            else: _board.id = 'RW'
            return _board
        
        c = move[0]
        j = int(move[1]) - 1

        # Loop through the column and set the last open spot with the color
        for i, spot in enumerate(_board[:, j]):
            if spot != 'O':
                _board[i - 1, j] = c
                break
            elif i == 5:
                _board[i, j] = c

        # Now we must figure out the id of this new board state
        # First check for horizontal wins
        for i in range(6):
            for j in range(4):
                if _board[i,j] == 'O': continue # Certainly isn't a win
                if _board[i,j] == _board[i,j+1] == _board[i,j+2] == _board[i,j+3]:
                    if _board[i,j] == 'R':
                        _board.id = 'RW'
                    else:
                        _board.id = 'BW'
                    return _board
        
        # Second check for vertical wins
        for i in range(3):
            for j in range(7):
                if _board[i,j] == 'O': continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j] == _board[i+2,j] == _board[i+3,j]:
                    if _board[i,j] == 'R':
                        _board.id = 'RW'
                    else:
                        _board.id = 'BW'
                    return _board
        
        # Third check for downwards diagonal wins
        for i in range(3):
            for j in range(4):
                if _board[i,j] == 'O': continue # Certainly isn't a win
                if _board[i,j] == _board[i+1,j+1] == _board[i+2,j+2] == _board[i+3,j+3]:
                    if _board[i,j] == 'R':
                        _board.id = 'RW'
                    else:
                        _board.id = 'BW'
                    return _board

        # Fourth check for upwards diagonal wins
        for i in range(3,6):
            for j in range(4):
                if _board[i,j] == 'O': continue # Certainly isn't a win
                if _board[i,j] == _board[i-1,j+1] == _board[i-2,j+2] == _board[i-3,j+3]:
                    if _board[i,j] == 'R':
                        _board.id = 'RW'
                    else:
                        _board.id = 'BW'
                    return _board

        # We have no connect 4's. Now check for a draw (no valid moves).
        if _board.moves == []:
            _board.id = 'D'

        # At this point, a valid piece placing move was made and the board is not in a terminal state.
        # So it's just the other players turn now.
        if _board.id == 'R': _board.id = 'B'
        else: _board.id = 'R'

        return _board
    
    def serialize(self) -> str:
        ser = self.id + ':'
        for i in range(6):
            for j in range(7):
                ser += self[i,j]
        return ser

    @classmethod
    def deserialize(cls, ser: str) -> 'connect_4_board':
        l = ser.split(':')

        _grid_str = l[1]
        _grid = np.full((6,7), 'O')
        _i = 0
        for i in range(6):
            for j in range(7):
                _grid[i,j] = _grid_str[_i]
                _i += 1
        
        return cls(grid = _grid, id = l[0])

class random_player(player):
    """
    This player makes random moves that are not among OD, DD, or R
    """
    def move(self, _board: connect_4_board) -> connect_4_board:
        _moves = _board.moves
        if 'OD' in _moves: _moves.remove('OD')
        if 'DD' in _moves: _moves.remove('DD')
        if 'R' in _moves: _moves.remove('R')
    
        _move = np.random.choice(_moves)

        return connect_4_board.move(_board, _move, False)

class cli_player(player):
    """
    This player plays from the command line interface, inputting moves as strings. 
    """
    def move(self, _board: connect_4_board) -> connect_4_board:
        print("The board currently looks like:")
        print(_board.grid)
        valid = False
        while not valid:
            print("The valid moves are:")
            print(_board.moves)
            _move = input("Please input a valid move: ")
            try:
                return _board.move(_board, _move, inplace = False)
            except InvalidMoveException:
                print("That was not a valid move.")
from __future__ import print_function
import numpy as np

def generate_knight_distance_matrix(game_size):
    size = (game_size * 2) - 1
    empty_matrix = np.array([[0] for i in range(size*size)]).reshape(size,size)

    start = (int(size/2), int(size/2))

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2),  (1, 2), (2, -1),  (2, 1)]
    iteration = 0
    empty_matrix[start] = -1
    full_matrix = fill_matrix(empty_matrix, directions, iteration)
    return full_matrix

def fill_matrix(curr_matrix, directions, iteration):
    dims = curr_matrix.shape

    if iteration == 0:
        moves = [(int(dims[0]/2),int(dims[0]/2))]
    else:
        locs = np.where(curr_matrix == iteration)
        moves = zip(locs[0],locs[1])
    for move in moves:
        for direction in directions:
            try_move = (move[0]+direction[0],move[1]+direction[1])
            if try_move[0] < 0 or try_move[0] >= dims[0] or try_move[1] < 0 or try_move[1] >= dims[1]:
                continue
            if curr_matrix[try_move] == 0:
                curr_matrix[try_move] = iteration+1
    if matrix_is_full(curr_matrix):
        return curr_matrix
    else:
        return fill_matrix(curr_matrix, directions, iteration+1)

def matrix_is_full(matrix):
    val_check = matrix != 0
    return np.all(val_check)

def get_moves_left_matrix(game):
    game_state_matrix = np.array([[0] for i in range(game.width*game.height)]).reshape(game.width,game.height)
    for move in game.get_blank_spaces():
        game_state_matrix[move] = 1
    return game_state_matrix

def get_sub_distance_matrix(location, size, distance_matrix):
    big_size = distance_matrix.shape[0]
    start_row = int(big_size/2) - location[0]
    end_row = start_row + size
    start_col = int(big_size/2) - location[1]
    end_col = start_col + size
    sub_matrix = distance_matrix[start_row:end_row,start_col:end_col]
    return sub_matrix


if __name__ == '__main__':
    generate_knight_distance_matrix(7)

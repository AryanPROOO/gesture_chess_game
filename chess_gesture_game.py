import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import chess
import random
import threading
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
BOARD_SIZE = 8
SQUARE_SIZE = 80
WIDTH = BOARD_SIZE * SQUARE_SIZE
HEIGHT = BOARD_SIZE * SQUARE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Chess - Complete Game")
clock = pygame.time.Clock()

# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
PIECE_WHITE = (255, 255, 255)
PIECE_BLACK = (50, 50, 50)

# Game state variables
board = chess.Board()
selected_square = None
locked_square = None
possible_moves = []
hover_square = None
long_press_start = 0
long_press_duration = 1.0
is_long_pressing = False
current_turn = "white"
game_over = False
winner = None

# Camera variables
camera_running = True
current_hand_pos = None
pinch_detected = False
gesture_lock = threading.Lock()

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def draw_piece_pawn(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Head
    pygame.draw.circle(surface, color, (center_x, center_y - 10), 12)
    # Body
    pygame.draw.rect(surface, color, (center_x - 6, center_y + 2, 12, 20))
    # Base
    pygame.draw.rect(surface, color, (center_x - 15, center_y + 22, 30, 6))

def draw_piece_rook(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Base
    pygame.draw.rect(surface, color, (center_x - 15, center_y - 15, 30, 35))
    # Battlements
    for i in range(5):
        if i % 2 == 0:
            pygame.draw.rect(surface, color, (center_x - 12 + i * 6, center_y - 25, 6, 10))

def draw_piece_knight(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Horse head shape
    points = [
        (center_x - 10, center_y + 15),
        (center_x - 15, center_y + 5),
        (center_x - 12, center_y - 10),
        (center_x - 5, center_y - 20),
        (center_x + 8, center_y - 15),
        (center_x + 15, center_y - 5),
        (center_x + 12, center_y + 5),
        (center_x + 8, center_y + 15)
    ]
    pygame.draw.polygon(surface, color, points)
    # Eye
    pygame.draw.circle(surface, WHITE if color == PIECE_BLACK else BLACK, (center_x - 3, center_y - 5), 3)

def draw_piece_bishop(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Mitre
    points = [
        (center_x - 12, center_y + 15),
        (center_x - 8, center_y - 15),
        (center_x + 8, center_y - 15),
        (center_x + 12, center_y + 15)
    ]
    pygame.draw.polygon(surface, color, points)
    # Cross
    pygame.draw.line(surface, WHITE if color == PIECE_BLACK else BLACK, 
                    (center_x - 5, center_y - 20), (center_x + 5, center_y - 20), 3)
    pygame.draw.line(surface, WHITE if color == PIECE_BLACK else BLACK, 
                    (center_x, center_y - 25), (center_x, center_y - 15), 3)

def draw_piece_queen(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Base crown
    pygame.draw.ellipse(surface, color, (center_x - 18, center_y - 5, 36, 25))
    # Crown spikes
    for i in range(-2, 3):
        spike_x = center_x + i * 8
        spike_height = 25 if abs(i) == 1 else 20
        pygame.draw.polygon(surface, color, [
            (spike_x - 3, center_y - 5),
            (spike_x, center_y - spike_height),
            (spike_x + 3, center_y - 5)
        ])

def draw_piece_king(surface, x, y, color):
    center_x, center_y = x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2
    # Base crown
    pygame.draw.ellipse(surface, color, (center_x - 16, center_y - 5, 32, 25))
    # Crown spikes
    for i in range(-1, 2):
        spike_x = center_x + i * 10
        pygame.draw.polygon(surface, color, [
            (spike_x - 4, center_y - 5),
            (spike_x, center_y - 20),
            (spike_x + 4, center_y - 5)
        ])
    # Cross on top
    pygame.draw.line(surface, WHITE if color == PIECE_BLACK else BLACK,
                    (center_x - 6, center_y - 25), (center_x + 6, center_y - 25), 3)
    pygame.draw.line(surface, WHITE if color == PIECE_BLACK else BLACK,
                    (center_x, center_y - 30), (center_x, center_y - 20), 3)

def draw_chess_piece(surface, piece, x, y):
    if not piece:
        return
    
    color = PIECE_WHITE if piece.color else PIECE_BLACK
    piece_type = piece.piece_type
    
    # Draw piece shadow
    shadow_offset = 2
    shadow_color = (100, 100, 100)
    
    if piece_type == chess.PAWN:
        draw_piece_pawn(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_pawn(surface, x, y, color)
    elif piece_type == chess.ROOK:
        draw_piece_rook(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_rook(surface, x, y, color)
    elif piece_type == chess.KNIGHT:
        draw_piece_knight(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_knight(surface, x, y, color)
    elif piece_type == chess.BISHOP:
        draw_piece_bishop(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_bishop(surface, x, y, color)
    elif piece_type == chess.QUEEN:
        draw_piece_queen(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_queen(surface, x, y, color)
    elif piece_type == chess.KING:
        draw_piece_king(surface, x + shadow_offset, y + shadow_offset, shadow_color)
        draw_piece_king(surface, x, y, color)

def square_to_coords(square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return file, 7 - rank

def coords_to_square(file, rank):
    return chess.square(file, 7 - rank)

def get_valid_moves(square):
    return [move.to_square for move in board.legal_moves if move.from_square == square]

def draw_board():
    screen.fill(BLACK)
    
    for rank in range(8):
        for file in range(8):
            color = WHITE if (file + rank) % 2 == 0 else BROWN
            rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            square = coords_to_square(file, rank)
            
            # Highlight squares
            if square == hover_square and not locked_square:
                pygame.draw.rect(screen, GREEN, rect, 6)
            elif square == locked_square:
                pygame.draw.rect(screen, RED, rect, 6)
            elif square in possible_moves:
                pygame.draw.circle(screen, YELLOW, rect.center, 15)
                if board.piece_at(square):  # Capture move
                    pygame.draw.circle(screen, RED, rect.center, 20, 5)
            
            # Draw piece
            piece = board.piece_at(square)
            if piece:
                draw_chess_piece(screen, piece, file * SQUARE_SIZE, rank * SQUARE_SIZE)
    
    # Draw coordinates
    font = pygame.font.Font(None, 24)
    for i in range(8):
        # Files (a-h)
        text = font.render(chr(ord('a') + i), True, BLACK)
        screen.blit(text, (i * SQUARE_SIZE + 5, HEIGHT - 20))
        # Ranks (1-8)
        text = font.render(str(8 - i), True, BLACK)
        screen.blit(text, (5, i * SQUARE_SIZE + 5))

def detect_pinch(hand_landmarks, frame_width, frame_height):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_x = int(thumb_tip.x * frame_width)
    thumb_y = int(thumb_tip.y * frame_height)
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)
    
    distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
    return distance < 30, distance

def camera_thread():
    global current_hand_pos, pinch_detected, camera_running
    
    cv2.namedWindow('Chess Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Chess Hand Tracking', 640, 480)
    cv2.moveWindow('Chess Hand Tracking', WIDTH + 50, 100)
    
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    grid_size = 8
    cell_width = cam_width // grid_size
    cell_height = cam_height // grid_size
    
    pinch_history = deque(maxlen=5)
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw grid
        for i in range(1, grid_size):
            cv2.line(frame, (i * cell_width, 0), (i * cell_width, cam_height), (0, 255, 0), 1)
            cv2.line(frame, (0, i * cell_height), (cam_width, i * cell_height), (0, 255, 0), 1)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get index finger position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * cam_width)
                y = int(index_tip.y * cam_height)
                
                grid_x = min(x // cell_width, grid_size - 1)
                grid_y = min(y // cell_height, grid_size - 1)
                
                with gesture_lock:
                    current_hand_pos = (grid_x, grid_y)
                
                # Detect pinch
                is_pinch, distance = detect_pinch(hand_landmarks, cam_width, cam_height)
                pinch_history.append(is_pinch)
                
                # Smooth pinch detection
                pinch_detected = sum(pinch_history) >= 3
                
                # Visual feedback
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
                # Highlight current grid
                top_left = (grid_x * cell_width, grid_y * cell_height)
                bottom_right = ((grid_x + 1) * cell_width, (grid_y + 1) * cell_height)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 3)
                
                # Display info
                chess_pos = f"{chr(ord('a') + grid_x)}{8 - grid_y}"
                cv2.putText(frame, f"Position: {chess_pos}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Pinch: {'YES' if pinch_detected else 'NO'}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if pinch_detected else (0, 0, 255), 2)
                cv2.putText(frame, f"Distance: {distance:.0f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                status = "LOCKED" if locked_square else "HOVER"
                cv2.putText(frame, f"Status: {status}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) if locked_square else (0, 255, 0), 2)
        
        cv2.imshow('Chess Hand Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_running = False
            break
    
    cv2.destroyAllWindows()

def make_ai_move():
    """Simple AI that makes random valid moves"""
    legal_moves = list(board.legal_moves)
    if legal_moves:
        move = random.choice(legal_moves)
        board.push(move)
        return True
    return False

def main():
    global selected_square, locked_square, possible_moves, hover_square
    global long_press_start, is_long_pressing, current_turn, game_over, winner
    
    # Start camera thread
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    running = True
    ai_move_timer = 0
    
    print("🎮 Gesture Chess Game Started!")
    print("📹 Camera window should open alongside")
    print("🤏 Use pinch gesture to select and move pieces")
    print("✋ White pieces (bottom) - Your turn")
    print("🤖 Black pieces (top) - AI opponent")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    # Reset game
                    board.reset()
                    selected_square = None
                    locked_square = None
                    possible_moves = []
                    hover_square = None
                    current_turn = "white"
                    game_over = False
                    winner = None
        
        if not game_over:
            with gesture_lock:
                if current_hand_pos and current_turn == "white":
                    file, rank = current_hand_pos
                    square = coords_to_square(file, rank)
                    
                    if not locked_square:
                        hover_square = square
                        
                        # Select piece with pinch
                        if pinch_detected and not is_long_pressing:
                            piece = board.piece_at(square)
                            if piece and piece.color == chess.WHITE:
                                is_long_pressing = True
                                long_press_start = time.time()
                        
                        # Long press to lock
                        if is_long_pressing:
                            press_duration = time.time() - long_press_start
                            if press_duration >= long_press_duration:
                                locked_square = square
                                possible_moves = get_valid_moves(square)
                                is_long_pressing = False
                                hover_square = None
                                print(f"🔒 Piece locked at {chess.square_name(square)}")
                                print(f"💡 Valid moves: {[chess.square_name(sq) for sq in possible_moves]}")
                    
                    else:
                        # Move piece
                        if square in possible_moves and pinch_detected and not is_long_pressing:
                            is_long_pressing = True
                            long_press_start = time.time()
                        
                        if is_long_pressing and square in possible_moves:
                            press_duration = time.time() - long_press_start
                            if press_duration >= long_press_duration:
                                # Make move
                                move = chess.Move(locked_square, square)
                                if move in board.legal_moves:
                                    board.push(move)
                                    print(f"♟️ Moved from {chess.square_name(locked_square)} to {chess.square_name(square)}")
                                    current_turn = "black"
                                    ai_move_timer = time.time()
                                
                                # Reset selection
                                locked_square = None
                                possible_moves = []
                                is_long_pressing = False
                                hover_square = None
                    
                    # Cancel long press if not pinching
                    if not pinch_detected and is_long_pressing:
                        is_long_pressing = False
            
            # AI move after delay
            if current_turn == "black" and time.time() - ai_move_timer > 1.0:
                if make_ai_move():
                    print("🤖 AI made a move")
                    current_turn = "white"
            
            # Check game over
            if board.is_game_over():
                game_over = True
                if board.is_checkmate():
                    winner = "Black" if board.turn == chess.WHITE else "White"
                    print(f"🏆 {winner} wins by checkmate!")
                elif board.is_stalemate():
                    winner = "Draw"
                    print("🤝 Game ends in stalemate!")
                else:
                    winner = "Draw"
                    print("🤝 Game ends in draw!")
        
        # Draw everything
        draw_board()
        
        # UI Text
        turn_text = font.render(f"Turn: {current_turn.upper()}", True, BLACK)
        screen.blit(turn_text, (10, HEIGHT + 10))
        
        if locked_square:
            piece = board.piece_at(locked_square)
            if piece:
                piece_name = chess.piece_name(piece.piece_type).title()
                locked_text = small_font.render(f"Selected: {piece_name} at {chess.square_name(locked_square)}", True, RED)
                screen.blit(locked_text, (10, HEIGHT + 40))
        
        if is_long_pressing:
            progress = min((time.time() - long_press_start) / long_press_duration, 1.0)
            progress_text = small_font.render(f"Long Press: {progress*100:.0f}%", True, BLUE)
            screen.blit(progress_text, (10, HEIGHT + 70))
        
        if game_over:
            game_over_text = font.render(f"Game Over! {winner}", True, RED)
            screen.blit(game_over_text, (WIDTH//2 - 100, HEIGHT//2))
            restart_text = small_font.render("Press R to restart", True, BLACK)
            screen.blit(restart_text, (WIDTH//2 - 80, HEIGHT//2 + 40))
        
        # Instructions
        instructions = [
            "🤏 Pinch & hold 1s to select piece",
            "💡 Yellow circles = valid moves",
            "🎯 Pinch & hold on valid move to place",
            "Press Q on camera to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = small_font.render(instruction, True, BLACK)
            screen.blit(inst_text, (WIDTH - 300, 20 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    camera_running = False
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

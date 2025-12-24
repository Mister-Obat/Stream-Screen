import socket
import struct
import sys
import io
import os
import time
import threading
import json
import collections

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import ctypes
# Création d'un ID unique basé sur le nom du fichier du script
# Cela permet à chaque app (ex: 'StreamScreen.pyw', 'stream_receiver.py') d'avoir sa propre icône dans la barre des tâches
try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    myappid = f'obat.{script_name}.v1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

# --- CONFIG ---
DEFAULT_PORT = 5555
UDP_PORT = 5555

# Global State
latest_frame = None
latest_frame_seq = 0
frame_lock = threading.Lock()
running = True

# Stats for Nerd Graph
ping_history = collections.deque(maxlen=50)

def discover_server():
    """Listens for UDP beacon to find Server IP auto-magically."""
    print("[AUTO-DISCOVERY] Ecoute du réseau...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', UDP_PORT))
    sock.settimeout(10.0) # 10s timeout
    
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            msg = data.decode()
            if msg.startswith("STREAM_SERVER"):
                print(f"[DECOUVERT] Serveur trouvé à {addr[0]}")
                return addr[0]
    except socket.timeout:
        print("[AUTO-DISCOVERY] Timeout. Echec.")
        return None
    finally:
        sock.close()

def send_command(sock, cmd_dict):
    try:
        p = json.dumps(cmd_dict) + "\n"
        sock.sendall(p.encode())
    except: pass

def network_thread_func(sock):
    global latest_frame, latest_frame_seq, running
    
    print("[NET] Thread démarré (Mode FAST/OUT-OF-ORDER).")
    
    def recv_n(n):
        buf = b''
        while len(buf) < n:
            try:
                c = sock.recv(n - len(buf))
                if not c: return None
                buf += c
            except: return None
        return buf

    while running:
        try:
            # Header: [SeqID (4)][Size (4)] => 8 bytes
            h = recv_n(8)
            if not h: break
            
            seq_id, size = struct.unpack(">LL", h)
            
            # Body
            data = recv_n(size)
            if not data: break
            
            # LOGIC: DROP IF OLD
            # If we receive frame 100 but we already displayed frame 101, 
            # 100 is useless trash. Drop it.
            with frame_lock:
                if seq_id > latest_frame_seq:
                    latest_frame = data
                    latest_frame_seq = seq_id
                    # Update stats (Seq Gap?)
                else:
                    # Dropped outdated frame
                    pass
                    
        except Exception as e:
            print(f"[NET ERR] {e}")
            break

def main():
    global running, latest_frame, latest_frame_seq
    
    # 1. Discovery
    target_ip = None
    if len(sys.argv) > 1:
        target_ip = sys.argv[1]
    else:
        target_ip = discover_server()
        
    if not target_ip:
        print("Serveur non trouvé. Essayez de spécifier l'IP: python stream_receiver.py IP")
        return

    # 2. Connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Connect] Tentative de connexion vers {target_ip}...")
    connected = False
    for i in range(10): # Try 10 times
        try:
            sock.connect((target_ip, DEFAULT_PORT))
            connected = True
            print("[Connect] Succès!")
            break
        except Exception as e:
            print(f"[Connect] Echec ({i+1}/10): {e}")
            time.sleep(1.0)
    
    if not connected:
        print("Abandon.")
        return

    # 3. Pygame
    pygame.init()
    info = pygame.display.Info()
    w, h = info.current_w, info.current_h
    screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont(None, 40)
    
    # Start Net Thread
    t = threading.Thread(target=network_thread_func, args=(sock,))
    t.daemon = True
    t.start()
    
    # Params
    config = {"quality": 50, "fps": 60, "res": "720p"}
    show_menu = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_m: show_menu = not show_menu
                
                if show_menu:
                    # Inputs
                    update = False
                    if event.key == pygame.K_UP:
                        config['quality'] = min(100, config['quality'] + 5)
                        update = True
                    elif event.key == pygame.K_DOWN:
                        config['quality'] = max(5, config['quality'] - 5)
                        update = True
                    elif event.key == pygame.K_RIGHT:
                        config['fps'] = min(120, config['fps'] + 10)
                        update = True
                    elif event.key == pygame.K_LEFT:
                        config['fps'] = max(10, config['fps'] - 10)
                        update = True
                    elif event.key == pygame.K_r:
                        res_list = ["480p", "720p", "1080p"]
                        curr_idx = res_list.index(config['res'])
                        config['res'] = res_list[(curr_idx + 1) % 3]
                        update = True
                        
                    if update: send_command(sock, config)

        # Render
        data = None
        with frame_lock:
            data = latest_frame
        
        if data:
            try:
                img = pygame.image.load(io.BytesIO(data))
                screen.blit(img, (0,0))
            except: pass
            
        # Menu
        if show_menu:
            # Overlay
            panel = pygame.Surface((600, 300))
            panel.set_alpha(180)
            panel.fill((0,0,0))
            screen.blit(panel, (50, 50))
            
            lines = [
                f"--- ULTIMATE STREAM SETTINGS ---",
                f"Qualité: {config['quality']}% (Haut/Bas)",
                f"FPS Cible: {config['fps']} (Gauche/Droite)",
                f"Résolution: {config['res']} (R)",
                f"Dernière SeqID: {latest_frame_seq}"
            ]
            for i, l in enumerate(lines):
                txt = font.render(l, True, (0, 255, 255))
                screen.blit(txt, (70, 70 + i*50))

        pygame.display.flip()
        clock.tick(120)

    sock.close()
    pygame.quit()

if __name__ == "__main__":
    main()

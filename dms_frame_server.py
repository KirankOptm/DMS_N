#!/usr/bin/env python3
"""
DMS Frame Server - Receives and displays frames from DMS client (IMX93 board)

Usage:
    python dms_frame_server.py [--port 5000]

Run this on your PC, then start DMS on board with:
    python dmsv9_npu_backup.py --remote_server <your-pc-ip> --remote_port 5000
"""

import socket
import struct
import cv2
import numpy as np
import argparse
import time
from threading import Thread

class FrameReceiver:
    """TCP server that receives frames from DMS client"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.frame = None
        self.frame_count = 0
        self.fps = 0
        self.fps_start = time.time()
        self.fps_count = 0
        
    def start(self):
        """Start the server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.running = True
            print(f"[Server] Listening on {self.host}:{self.port}")
            print(f"[Server] Waiting for DMS client connection...")
            return True
        except Exception as e:
            print(f"[Server] Failed to start: {e}")
            return False
    
    def accept_client(self):
        """Accept a client connection"""
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"[Server] Client connected from {addr}")
            return True
        except Exception as e:
            print(f"[Server] Accept error: {e}")
            return False
    
    def receive_frame(self):
        """Receive a single frame from client"""
        try:
            # Receive frame size (4 bytes)
            size_data = self._recv_all(4)
            if not size_data:
                return None
            
            frame_size = struct.unpack('!I', size_data)[0]
            
            # Receive frame data
            frame_data = self._recv_all(frame_size)
            if not frame_data:
                return None
            
            # Decode JPEG
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Update FPS counter
            self.frame_count += 1
            self.fps_count += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.fps = self.fps_count / elapsed
                self.fps_count = 0
                self.fps_start = time.time()
            
            return frame
            
        except Exception as e:
            print(f"[Server] Receive error: {e}")
            return None
    
    def _recv_all(self, size):
        """Receive exactly 'size' bytes"""
        data = b''
        while len(data) < size:
            chunk = self.client_socket.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("[Server] Stopped")


def main():
    parser = argparse.ArgumentParser(description='DMS Frame Server - Display frames from DMS client')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    args = parser.parse_args()
    
    # Get local IP for display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print("=" * 60)
    print("DMS Frame Server")
    print("=" * 60)
    print(f"Local IP: {local_ip}")
    print(f"Port: {args.port}")
    print("")
    print("To connect DMS client (on IMX93 board), run:")
    print(f"  python dmsv9_npu_backup.py --remote_server {local_ip} --remote_port {args.port}")
    print("=" * 60)
    
    receiver = FrameReceiver(host=args.host, port=args.port)
    
    if not receiver.start():
        return
    
    # Main loop: accept clients and display frames
    while True:
        try:
            if not receiver.accept_client():
                continue
            
            print("[Server] Starting frame display...")
            cv2.namedWindow("DMS Remote View", cv2.WINDOW_NORMAL)
            
            while receiver.running:
                frame = receiver.receive_frame()
                
                if frame is None:
                    print("[Server] Client disconnected")
                    break
                
                # Add FPS overlay
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Recv FPS: {receiver.fps:.1f}", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {receiver.frame_count}", (10, h - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow("DMS Remote View", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("[Server] User pressed ESC, exiting...")
                    receiver.running = False
                    break
                elif key == ord('s'):  # Save screenshot
                    filename = f"dms_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[Server] Screenshot saved: {filename}")
            
            cv2.destroyAllWindows()
            
            if not receiver.running:
                break
                
            print("[Server] Waiting for new client...")
            
        except KeyboardInterrupt:
            print("\n[Server] Interrupted by user")
            break
    
    receiver.stop()
    print("[Server] Goodbye!")


if __name__ == "__main__":
    main()

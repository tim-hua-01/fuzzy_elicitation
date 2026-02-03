#!/usr/bin/env python3
"""
Simple HTTP server for the experiment viewer.
Run this and open http://localhost:8000/viewer.html
"""

import http.server
import json
import os
import socket
import socketserver
from pathlib import Path

PORT = 8000

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API endpoint for manifest by rubric version
        if self.path.startswith('/api/manifest/'):
            version = self.path.split('/')[-1]
            base_dir = Path(__file__).parent / 'data' / version

            runs = []
            human_grades = []

            runs_dir = base_dir / 'runs'
            if runs_dir.exists():
                runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

            human_dir = base_dir / 'human_grades'
            if human_dir.exists():
                human_grades = sorted([f.name for f in human_dir.iterdir() if f.suffix == '.jsonl'])

            manifest = {
                'runs': runs,
                'human_grades': human_grades
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(manifest).encode())
            return

        return super().do_GET()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)

    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        local_ip = get_local_ip()
        print(f"Server started on port {PORT}")
        print(f"\nLocal access:")
        print(f"  http://localhost:{PORT}/viewer.html")
        print(f"\nNetwork access (same LAN):")
        print(f"  http://{local_ip}:{PORT}/viewer.html")
        print(f"\nFor external access (internet), use one of these options:")
        print(f"  1. Cloudflared tunnel: cloudflared tunnel --url http://localhost:{PORT}")
        print(f"  2. ngrok: ngrok http {PORT}")
        print(f"  3. SSH tunnel: ssh -L {PORT}:localhost:{PORT} user@this-machine")
        print("\nPress Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")

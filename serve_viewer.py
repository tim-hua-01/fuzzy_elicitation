#!/usr/bin/env python3
"""
Simple HTTP server for the experiment viewer.
Run this and open http://localhost:8000/viewer.html

Usage:
    python serve_viewer.py           # Binds to all interfaces (LAN accessible)
    python serve_viewer.py --local   # Binds to localhost only (more secure)
"""

import argparse
import http.server
import json
import os
import socket
import socketserver
from pathlib import Path
from urllib.parse import unquote

PORT = 8000
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
ALLOWED_FILES = {
    '/viewer.html': BASE_DIR / 'viewer.html',
    '/blind_eval.html': BASE_DIR / 'blind_eval.html',
    '/main_questions.jsonl': BASE_DIR / 'main_questions.jsonl',
}


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Decode URL-encoded path
        path = unquote(self.path.split('?')[0])
        
        # API endpoint for manifest by rubric version
        if path.startswith('/api/manifest/'):
            version = path.split('/')[-1]
            base_dir = DATA_DIR / version
            
            # Validate version doesn't escape data directory
            try:
                base_dir.resolve().relative_to(DATA_DIR.resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return

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

        # API endpoint for available graders for a specific run
        if path.startswith('/api/graders/'):
            parts = path.split('/')
            if len(parts) >= 5:
                version = parts[3]
                run_name = parts[4]
                run_dir = DATA_DIR / version / 'runs' / run_name
                
                # Validate path doesn't escape data directory
                try:
                    run_dir.resolve().relative_to(DATA_DIR.resolve())
                except ValueError:
                    self.send_error(403, "Forbidden")
                    return

                graders = []
                other_graders = []
                if run_dir.exists():
                    # Find all grades files
                    for f in run_dir.iterdir():
                        if f.name.startswith('grades') and f.suffix == '.jsonl':
                            if f.name == 'grades.jsonl':
                                # Default grader goes first
                                graders.append({'name': 'default', 'file': 'grades.jsonl'})
                            else:
                                # Extract model name from grades_{model}.jsonl
                                model_name = f.name[7:-6]  # Remove 'grades_' and '.jsonl'
                                other_graders.append({'name': model_name, 'file': f.name})
                
                # Sort other graders alphabetically and append after default
                other_graders.sort(key=lambda x: x['name'])
                graders.extend(other_graders)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(graders).encode())
                return
            
            self.send_error(400, "Bad Request")
            return

        # Serve explicitly allowed files (viewer.html)
        if path in ALLOWED_FILES:
            file_path = ALLOWED_FILES[path]
            if file_path.exists():
                self._serve_file(file_path)
                return
            self.send_error(404, "Not Found")
            return
        
        # Serve files only from data/ directory
        if path.startswith('/data/'):
            relative_path = path[6:]  # Remove '/data/' prefix
            file_path = DATA_DIR / relative_path
            
            # Resolve and validate path stays within data directory
            try:
                resolved = file_path.resolve()
                resolved.relative_to(DATA_DIR.resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return
            
            if resolved.is_file():
                self._serve_file(resolved)
                return
            
            self.send_error(404, "Not Found")
            return
        
        # Redirect root to viewer
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/viewer.html')
            self.end_headers()
            return
        
        # Block everything else
        self.send_error(403, "Forbidden - only /viewer.html and /data/ are accessible")
    
    def _serve_file(self, file_path: Path):
        """Serve a file with appropriate content type."""
        content_types = {
            '.html': 'text/html',
            '.json': 'application/json',
            '.jsonl': 'application/jsonl',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.js': 'application/javascript',
            '.css': 'text/css',
        }
        ext = file_path.suffix.lower()
        content_type = content_types.get(ext, 'application/octet-stream')
        
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-Length', len(content))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")

    def log_message(self, format, *args):
        # Quieter logging - only show the request
        print(f"{self.address_string()} - {args[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serve the experiment viewer')
    parser.add_argument('--local', action='store_true', 
                        help='Bind to localhost only (prevents LAN/internet access)')
    parser.add_argument('--port', type=int, default=PORT,
                        help=f'Port to serve on (default: {PORT})')
    args = parser.parse_args()
    
    bind_address = "127.0.0.1" if args.local else "0.0.0.0"
    port = args.port

    with socketserver.TCPServer((bind_address, port), Handler) as httpd:
        print(f"Server started on port {port}")
        print(f"Serving only: /viewer.html, /blind_eval.html, and /data/*")
        print(f"\nLocal access:")
        print(f"  http://localhost:{port}/viewer.html")
        print(f"  http://localhost:{port}/blind_eval.html")
        
        if not args.local:
            local_ip = get_local_ip()
            print(f"\nNetwork access (same LAN):")
            print(f"  http://{local_ip}:{port}/viewer.html")
            print(f"\nFor external access (internet), use one of these options:")
            print(f"  1. Cloudflared tunnel: cloudflared tunnel --url http://localhost:{port}")
            print(f"  2. ngrok: ngrok http {port}")
            print(f"  3. SSH tunnel: ssh -L {port}:localhost:{port} user@this-machine")
        else:
            print(f"\n[--local mode] Server is only accessible from this machine")
        
        print("\nPress Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")

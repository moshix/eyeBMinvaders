#!/usr/bin/env python3
"""
Game server with gameplay recording endpoint.

Serves the game files AND provides /api/gameplay POST endpoint
that saves recorded transitions to models/gameplay_data.jsonl.

Usage:
    python serve.py              # port 8080
    python serve.py --port 9000  # custom port
"""

import http.server
import json
import os
import sys
import time
from urllib.parse import urlparse

GAMEPLAY_FILE = os.path.join(os.path.dirname(__file__), 'models', 'gameplay_data.jsonl')

class GameHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/gameplay':
            self._handle_gameplay_post()
        else:
            self.send_error(404)

    def _handle_gameplay_post(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            transitions = data.get('transitions', [])
            episodes = data.get('episodes', 0)
            timestamp = data.get('timestamp', '')

            # Append each transition as a JSON line
            os.makedirs(os.path.dirname(GAMEPLAY_FILE), exist_ok=True)
            with open(GAMEPLAY_FILE, 'a') as f:
                for t in transitions:
                    f.write(json.dumps(t) + '\n')

            total_lines = 0
            if os.path.exists(GAMEPLAY_FILE):
                with open(GAMEPLAY_FILE) as f:
                    total_lines = sum(1 for _ in f)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'saved': len(transitions),
                'total': total_lines,
                'file': GAMEPLAY_FILE,
            }).encode())

            print(f"[gameplay] +{len(transitions)} transitions (total: {total_lines}, ep: {episodes})")

        except Exception as e:
            self.send_error(500, str(e))

    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress static file logs, only show API calls
        if '/api/' in (args[0] if args else ''):
            super().log_message(format, *args)

if __name__ == '__main__':
    port = 8080
    if '--port' in sys.argv:
        port = int(sys.argv[sys.argv.index('--port') + 1])

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.HTTPServer(('', port), GameHandler)
    print(f"Serving eyeBMinvaders on http://localhost:{port}")
    print(f"Gameplay recording saves to: {GAMEPLAY_FILE}")
    print(f"Press W in game to start recording, play, data auto-saves every 5000 steps")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

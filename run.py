#!/usr/bin/env python3
"""
Simple HTTP server to run the Circle Detection web application.

Usage:
    python run.py [port]

Default port is 8000 if not specified.
"""

import http.server
import socketserver
import sys
import os
import webbrowser
from pathlib import Path


def main():
    # Get port from command line argument or use default
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            print("Usage: python run.py [port]")
            sys.exit(1)

    # Change to public directory where web files are located
    script_dir = Path(__file__).parent.absolute()
    public_dir = script_dir / "public"
    os.chdir(public_dir)

    # Create HTTP server with standard handler
    Handler = http.server.SimpleHTTPRequestHandler

    # Allow address reuse to prevent "Address already in use" errors
    socketserver.TCPServer.allow_reuse_address = True

    try:
        httpd = socketserver.TCPServer(("", port), Handler)

        url = f"http://localhost:{port}"
        print("=" * 60)
        print("🎯 Circle & Block Detection Web Application")
        print("=" * 60)
        print(f"Server running at: {url}")
        print(f"Directory: {public_dir}")
        print("\n📂 Available files:")
        print(f"   • {url}/index.html (Main application)")
        print(f"   • {url}/processor.py (Processing module)")
        print(f"   • {url}/README.md (Documentation)")
        print("\n💡 Tips:")
        print("   • Open index.html in your browser")
        print("   • Wait for PyScript to initialize")
        print("   • Upload your TIFF file and process")
        print("\n⌨️  Press Ctrl+C to stop the server")
        print("=" * 60)
        print()

        # Try to open browser automatically
        try:
            webbrowser.open(f"{url}/image-analysis-miu-batubara/index.html")
            print(f"✅ Opening {url}/image-analysis-miu-batubara/index.html in your browser...")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Please manually open: {url}/image-analysis-miu-batubara/index.html")

        print()

        # Start serving - serve_forever() handles Ctrl+C better
        httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("Goodbye! 👋")
        sys.exit(0)
    except OSError as e:
        if e.errno == 98 or e.errno == 10048:  # Address already in use
            print(f"\n❌ Error: Port {port} is already in use")
            print(f"   Try a different port: python run.py {port + 1}")
        else:
            print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

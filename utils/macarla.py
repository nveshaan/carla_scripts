#!/usr/bin/env python

"""
Simple Pyro server for CARLA remote access
"""

import Pyro5.api
import Pyro5.server
import carla
import logging
import threading
import time
import sys
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@Pyro5.api.expose
class CarlaService:
    """Simple CARLA service using Pyro"""
    
    def __init__(self):
        self._clients = {}
        self._client_lock = threading.Lock()
        logger.info("CarlaService initialized")
    
    def carla(self):
        """Expose the carla module directly"""
        return carla
    
    def ping(self):
        """Simple ping to test connection"""
        return "pong"

def main():
    port = 18861
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    logger.info(f"Starting CARLA Pyro server on port {port}")
    
    try:
        # Start the Pyro daemon
        daemon = Pyro5.server.Daemon(port=port)
        
        # Register the CarlaService
        service = CarlaService()
        uri = daemon.register(service)
        
        logger.info(f"Service registered with URI: {uri}")
        logger.info("Server started successfully")
        logger.info("Press Ctrl+C to stop the server")
        
        # Start the daemon request loop
        daemon.requestLoop()
        
    except KeyboardInterrupt:
        logger.info("Server stopping...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

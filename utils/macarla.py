#!/usr/bin/env python

"""
Improved rpyc server for CARLA remote access
"""

import rpyc
import carla
import logging
import threading
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarlaService(rpyc.Service):
    """Enhanced CARLA service with better error handling and connection management"""
    
    def __init__(self):
        super().__init__()
        self._clients = {}
        self._client_lock = threading.Lock()
        logger.info("CarlaService initialized")
    
    def exposed_carla(self):
        """Expose the carla module"""
        return carla
    
    def exposed_get_client(self, host='127.0.0.1', port=2000, timeout=10.0):
        """Get a CARLA client with connection caching"""
        client_key = f"{host}:{port}"
        
        with self._client_lock:
            if client_key not in self._clients:
                try:
                    client = carla.Client(host, port)
                    client.set_timeout(timeout)
                    # Test connection
                    world = client.get_world()
                    self._clients[client_key] = client
                    logger.info(f"Created new client for {client_key}")
                except Exception as e:
                    logger.error(f"Failed to create client for {client_key}: {e}")
                    raise
            
            return self._clients[client_key]
    
    def exposed_get_world(self, host='127.0.0.1', port=2000):
        """Get the CARLA world"""
        client = self.exposed_get_client(host, port)
        return client.get_world()
    
    def exposed_ping(self):
        """Simple ping to test connection"""
        return "pong"
    
    def on_connect(self, conn):
        """Called when a client connects"""
        logger.info(f"Client connected: {conn}")
    
    def on_disconnect(self, conn):
        """Called when a client disconnects"""
        logger.info(f"Client disconnected: {conn}")

def main():
    port = 18861
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    # Enhanced protocol configuration
    protocol_config = {
        "allow_public_attrs": True,
        "allow_setattr": True,
        "allow_getattr": True,
        "allow_delattr": True,
        "allow_pickle": True,
        "sync_request_timeout": 10,  # Shorter timeout
        "max_serialization_depth": 5,  # Reduced depth
        "allow_all_attrs": True,
        "instantiate_custom_exceptions": True,
        "import_custom_exceptions": True,
        "connid_to_stream_timeout": 10,
        "stream_chunk_size": 8192,  # Smaller chunks
    }
    
    logger.info(f"Starting CARLA rpyc server on port {port}")
    logger.info("Protocol configuration:")
    for key, value in protocol_config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Use ThreadedServer for better handling of multiple clients
        from rpyc.utils.server import ThreadedServer
        server = ThreadedServer(
            CarlaService,
            port=port,
            protocol_config=protocol_config,
            logger=logger
        )
        
        logger.info("Server started successfully")
        logger.info("Press Ctrl+C to stop the server")
        
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Server stopping...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

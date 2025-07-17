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

class FilteredWorld:
    """Wrapper for carla.World that filters spawn_actor calls"""
    
    def __init__(self, world):
        self._world = world
        self._blocked_sensors = {
            'sensor.camera.rgb',
            'sensor.camera.depth',
            'sensor.camera.semantic_segmentation',
            'sensor.lidar.ray_cast',
            'sensor.other.gnss',
            'sensor.other.imu',
            'sensor.other.collision',
            'sensor.other.lane_invasion',
            'sensor.other.obstacle',
            'sensor.other.radar'
        }
    
    def spawn_actor(self, blueprint, transform, attach_to=None):
        """Filter sensor spawns"""
        if hasattr(blueprint, 'id') and blueprint.id in self._blocked_sensors:
            logger.warning(f"Blocked sensor spawn: {blueprint.id}")
            return None
        
        # Allow non-sensor actors
        return self._world.spawn_actor(blueprint, transform, attach_to)
    
    def try_spawn_actor(self, blueprint, transform, attach_to=None):
        """Filter sensor spawns for try_spawn_actor"""
        if hasattr(blueprint, 'id') and blueprint.id in self._blocked_sensors:
            logger.warning(f"Blocked sensor try_spawn: {blueprint.id}")
            return None
        
        return self._world.try_spawn_actor(blueprint, transform, attach_to)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the real world"""
        return getattr(self._world, name)

class FilteredClient:
    """Wrapper for carla.Client that returns filtered world"""
    
    def __init__(self, client):
        self._client = client
    
    def get_world(self):
        """Return a filtered world"""
        world = self._client.get_world()
        return FilteredWorld(world)
    
    def load_world(self, map_name):
        """Load world and return filtered version"""
        world = self._client.load_world(map_name)
        return FilteredWorld(world)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the real client"""
        return getattr(self._client, name)

class CarlaService(rpyc.Service):
    """Enhanced CARLA service with better error handling and connection management"""
    
    def __init__(self):
        super().__init__()
        self._clients = {}
        self._client_lock = threading.Lock()
        logger.info("CarlaService initialized")
    
    def exposed_carla(self):
        """Expose the carla module with filtered Client"""
        # Create a module-like object
        class FilteredCarlaModule:
            def Client(self, host='localhost', port=2000, worker_threads=0):
                client = carla.Client(host, port, worker_threads)
                return FilteredClient(client)
            
            def __getattr__(self, name):
                return getattr(carla, name)
        
        return FilteredCarlaModule()
    
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

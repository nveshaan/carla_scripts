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
import zmq
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensorDataPublisher:
    """ZeroMQ publisher for sensor data"""
    
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"Sensor data publisher started on port {port}")
    
    def publish_sensor_data(self, sensor_type, data):
        """Publish sensor data over ZeroMQ"""
        try:
            message = {
                'sensor_type': sensor_type,
                'timestamp': time.time(),
                'data': data
            }
            # Use sensor_type as topic for filtering
            self.socket.send_multipart([
                sensor_type.encode('utf-8'),
                pickle.dumps(message)
            ])
            logger.debug(f"Published {sensor_type} data")
        except Exception as e:
            logger.error(f"Failed to publish sensor data: {e}")
    
    def close(self):
        """Clean up publisher"""
        self.socket.close()
        self.context.term()

# Global publisher instance
sensor_publisher = SensorDataPublisher()

class FilteredActor:
    """Wrapper for sensor actors that publishes sensor data over ZeroMQ"""
    
    def __init__(self, actor, sensor_type):
        self._actor = actor
        self._sensor_type = sensor_type
        self._original_listen = actor.listen
    
    def listen(self, callback):
        """Intercept sensor listen calls and publish data over ZeroMQ"""
        logger.info(f"Intercepting sensor listen for: {self._sensor_type}")
        
        def zmq_callback(data):
            """Callback that publishes to ZeroMQ and calls original callback"""
            try:
                # Publish sensor data over ZeroMQ
                sensor_publisher.publish_sensor_data(self._sensor_type, data)
                
                # Also call the original callback if needed
                if callback:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in ZMQ callback for {self._sensor_type}: {e}")
        
        # Use the original listen method with our custom callback
        return self._original_listen(zmq_callback)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the real actor"""
        return getattr(self._actor, name)

class FilteredWorld:
    """Wrapper for carla.World that returns sensor actors with ZeroMQ publishing"""
    
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
        """Spawn actor and return filtered version if it's a sensor"""
        actor = self._world.spawn_actor(blueprint, transform, attach_to)
        
        # If it's a sensor, return a filtered version
        if actor and hasattr(blueprint, 'id') and blueprint.id in self._blocked_sensors:
            return FilteredActor(actor, blueprint.id)
        
        return actor
    
    def try_spawn_actor(self, blueprint, transform, attach_to=None):
        """Try to spawn actor and return filtered version if it's a sensor"""
        actor = self._world.try_spawn_actor(blueprint, transform, attach_to)
        
        # If it's a sensor, return a filtered version
        if actor and hasattr(blueprint, 'id') and blueprint.id in self._blocked_sensors:
            return FilteredActor(actor, blueprint.id)
        
        return actor
    
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
        sensor_publisher.close()
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sensor_publisher.close()

if __name__ == '__main__':
    main()

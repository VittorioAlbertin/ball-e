"""
People Database Service Node - SERVER
File location: ros2_ws/src/perception_pkg/perception_pkg/people_database_node.py

This node runs continuously and provides database services that other nodes can call.
It manages all database operations for people recognition.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import json

# Import service definitions from msgs_interfaces
from msgs_interfaces.srv import (
    AddPerson, RecognizeFace, GetPerson, UpdateLastSeen,
    UpdatePreferences, GetAllPeople, DeletePerson
)

from interaction_pkg.interaction_pkg.people_database import PeopleDatabase


class PeopleDatabaseNode(Node):
    """
    ROS2 Service node that provides database operations for people management.
    This is the SERVER - it runs continuously and responds to service calls.
    """
    
    def __init__(self):
        super().__init__('people_database_node')
        
        # Declare parameter for database path
        self.declare_parameter('db_path', '/ros2_ws/robot_data/people.db')
        db_path = self.get_parameter('db_path').value
        
        # Initialize database
        self.db = PeopleDatabase(db_path)
        self.get_logger().info(f'People database initialized at {db_path}')
        
        # Create services (SERVER endpoints)
        self.srv_add_person = self.create_service(
            AddPerson, 'people_db/add_person', self.add_person_callback)
        
        self.srv_recognize_face = self.create_service(
            RecognizeFace, 'people_db/recognize_face', self.recognize_face_callback)
        
        self.srv_get_person = self.create_service(
            GetPerson, 'people_db/get_person', self.get_person_callback)
        
        self.srv_update_last_seen = self.create_service(
            UpdateLastSeen, 'people_db/update_last_seen', self.update_last_seen_callback)
        
        self.srv_update_preferences = self.create_service(
            UpdatePreferences, 'people_db/update_preferences', self.update_preferences_callback)
        
        self.srv_get_all_people = self.create_service(
            GetAllPeople, 'people_db/get_all_people', self.get_all_people_callback)
        
        self.srv_delete_person = self.create_service(
            DeletePerson, 'people_db/delete_person', self.delete_person_callback)
        
        self.get_logger().info('People database services ready! Other nodes can now call them.')
    
    def add_person_callback(self, request, response):
        """Service callback: Add a new person to the database."""
        try:
            embedding = np.array(request.face_embedding, dtype=np.float32)
            
            # Parse preferences if provided
            preferences = None
            if request.preferences_json:
                preferences = json.loads(request.preferences_json)
            
            person_id = self.db.add_person(
                name=request.name,
                face_embedding=embedding,
                preferences=preferences,
                notes=request.notes if request.notes else None
            )
            
            response.success = True
            response.person_id = person_id
            response.message = f"Successfully added {request.name} with ID {person_id}"
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.person_id = -1
            response.message = f"Error adding person: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def recognize_face_callback(self, request, response):
        """Service callback: Find a person by face embedding similarity."""
        try:
            embedding = np.array(request.face_embedding, dtype=np.float32)
            threshold = request.threshold if request.threshold > 0 else 0.6
            
            match = self.db.find_similar_face(embedding, threshold)
            
            if match:
                response.found = True
                response.person_id = match['id']
                response.name = match['name']
                response.last_seen = str(match['last_seen'])
                response.interaction_count = match['interaction_count']
                response.preferences_json = json.dumps(match['preferences']) if match['preferences'] else ""
                response.notes = match['notes'] if match['notes'] else ""
                response.message = f"Recognized {match['name']}"
                
                # Auto-update last seen
                self.db.update_last_seen(match['id'])
                self.get_logger().info(f"Recognized: {match['name']} (ID: {match['id']})")
            else:
                response.found = False
                response.message = "No matching person found"
                self.get_logger().info("Face not recognized")
                
        except Exception as e:
            response.found = False
            response.message = f"Error during recognition: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def get_person_callback(self, request, response):
        """Service callback: Get person info by ID or name."""
        try:
            person = None
            
            if request.person_id >= 0:
                person = self.db.get_person_by_id(request.person_id)
            elif request.name:
                person = self.db.get_person_by_name(request.name)
            
            if person:
                response.found = True
                response.person_id = person['id']
                response.name = person['name']
                response.last_seen = str(person['last_seen'])
                response.created_at = str(person['created_at'])
                response.interaction_count = person['interaction_count']
                response.preferences_json = json.dumps(person['preferences']) if person['preferences'] else ""
                response.notes = person['notes'] if person['notes'] else ""
                response.message = "Person found"
            else:
                response.found = False
                response.message = "Person not found"
                
        except Exception as e:
            response.found = False
            response.message = f"Error retrieving person: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def update_last_seen_callback(self, request, response):
        """Service callback: Update last seen timestamp."""
        try:
            self.db.update_last_seen(request.person_id)
            response.success = True
            response.message = f"Updated last seen for person ID {request.person_id}"
            
        except Exception as e:
            response.success = False
            response.message = f"Error updating last seen: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def update_preferences_callback(self, request, response):
        """Service callback: Update user preferences."""
        try:
            preferences = json.loads(request.preferences_json)
            self.db.update_preferences(request.person_id, preferences)
            response.success = True
            response.message = f"Updated preferences for person ID {request.person_id}"
            
        except Exception as e:
            response.success = False
            response.message = f"Error updating preferences: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def get_all_people_callback(self, request, response):
        """Service callback: Get list of all people."""
        try:
            all_people = self.db.get_all_people()
            
            response.person_ids = [p['id'] for p in all_people]
            response.names = [p['name'] for p in all_people]
            response.last_seen_times = [str(p['last_seen']) for p in all_people]
            response.interaction_counts = [p['interaction_count'] for p in all_people]
            
            self.get_logger().info(f"Retrieved {len(all_people)} people")
            
        except Exception as e:
            self.get_logger().error(f"Error retrieving all people: {str(e)}")
        
        return response
    
    def delete_person_callback(self, request, response):
        """Service callback: Delete a person from the database."""
        try:
            self.db.delete_person(request.person_id)
            response.success = True
            response.message = f"Deleted person ID {request.person_id}"
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f"Error deleting person: {str(e)}"
            self.get_logger().error(response.message)
        
        return response
    
    def destroy_node(self):
        """Clean up database connection on shutdown."""
        self.get_logger().info('Shutting down people database node')
        self.db.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PeopleDatabaseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
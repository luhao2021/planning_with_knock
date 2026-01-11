#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from blocksworld_scene.srv import UpdateScene
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.srv import ApplyPlanningScene


import subprocess
import time
import re
import os
import trimesh
import argparse
import pathlib
import yaml
import tf_transformations as tft

class GazeboPoseUpdater(Node):
    def __init__(self, config, ready_flag):
        super().__init__('gazebo_pose_updater')
        self.config = config
        # Load commands from config.
        self.availability_check_cmd = self.config["availability_check_service"]["command"].split() + \
                                        self.config["availability_check_service"]["parameters"].split()
        self.initial_scene_layout_cmd = self.config["initial_scene_layout_service"]["command"].split() + \
                                        ["-s", self.config["initial_scene_layout_service"]["service_name"]] + \
                                        self.config["initial_scene_layout_service"]["parameters"].split()
        self.pose_update_cmd = self.config["pose_update_topic"]["command"].split() + \
                               ["-t", self.config["pose_update_topic"]["topic_name"]] + \
                               self.config["pose_update_topic"]["parameters"].split()
        self.objects_of_interest = set(self.config["objects_of_interest"])
        self.z_base = self.config["z_base"]
        self.z_inc = self.config["z_increment"]

        self.ready_flag = ready_flag
        
        print("availability check command:", self.availability_check_cmd)
        print("initial scene info service command:", self.initial_scene_layout_cmd)
        print("pose update command:", self.pose_update_cmd)
        print("objects of interests:", self.objects_of_interest)

        self.get_logger().info("Initializing GazeboPoseUpdater from config...")
        # Publisher for updating the planning scene
        self.planning_scene_pub = self.create_publisher(PlanningScene, 'planning_scene', 10)
        # Call the initial service to get the scene layout.
        self.initial_scene_text = self.call_ign_service()
        self.objects = self.parse_scene(self.initial_scene_text)
        # Here you might also build and store your complete CollisionObjects for each object.
        # For this example, we simply store the parsed object pose information.
        self.get_logger().info(f"Initialized with {len(self.objects)} objects from scene.")

        if not self.objects:
            self.get_logger().warn("No models found in the scene info")
            return

        for obj in self.objects:
            print("Object:", obj["name"])
            print("  Pose:", obj["pose"])
            if "geometry" in obj:
                print("  Geometry:", obj["geometry"])
            if "material" in obj:
                print("  Material:", obj["material"])
            print()

        # Create a planning scene update message.
        planning_scene_msg = PlanningScene()
        planning_scene_msg.is_diff = True

        # For each model of interest, create a CollisionObject.
        for model in self.objects:
            if self.match_interests(model.get('name')):
                collision_object, color = self.create_collision_object(model)
                planning_scene_msg.world.collision_objects.append(collision_object)
                print(color)
                planning_scene_msg.object_colors.append(color)
                self.get_logger().info(f"Updated collision object: {model.get('name')}")

        # Publish the planning scene update.
        self.planning_scene_pub.publish(planning_scene_msg)

        self.srv = self.create_service(Trigger, '/update_smash_scene', self.scene_update)
        self.srv = self.create_service(Trigger, '/remove_smash_scene_objects', self.scene_remove_objects)
        self.srv = self.create_service(Trigger, "/add_smash_scene_objects", self.scene_add_objects)
        self.srv = self.create_service(UpdateScene, "/update_smash_scene_objects", self.scene_update_objects)
        # Create a periodic timer to update poses (using the pose update topic command).
        #self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz update
        self.scene_objects_deleted = False

        # Call this once in your __init__()
        self.marker_pub = self.create_publisher(MarkerArray, '/smash_scene_markers', qos_profile=10)
        self.srv = self.create_service(Trigger, '/add_smash_scene_markers', self.scene_add_markers)
        self.srv = self.create_service(Trigger, '/remove_smash_scene_markers', self.scene_remove_markers)
        self.cli = self.create_client(ApplyPlanningScene, 'apply_planning_scene')


        if self.ready_flag:
            self.ready_flag.write_text("ready\n")
            self.get_logger().info("Mark READY!")

    def wait_for_ign_service(self, service_name, timeout_sec=300, poll_interval=1.0):
        """
        Waits until the given Ignition service is available.
        Checks both the service info and the list of services.
        """
        self.get_logger().info(f"Waiting for service {service_name} to become available...")
        start_time = time.time()
        error_message = f"No service providers on service [{service_name}]"
        while time.time() - start_time < timeout_sec:
            try:
                cmd_list = self.config["availability_check_service"]["command"].split() + \
                           self.config["availability_check_service"]["parameters"].split()
                output_list = subprocess.check_output(cmd_list, universal_newlines=True)
                if service_name in output_list:
                    self.get_logger().info(f"Service {service_name} is available (via service list).")
                    return True
            except Exception as e:
                self.get_logger().debug(f"Service list check failed: {e}")

            time.sleep(poll_interval)
        self.get_logger().error(f"Timeout waiting for service {service_name}.")
        return False

    def call_ign_service(self):
        """
        Waits for the initial scene service to become available, then calls it.
        """
        service_name = self.config["initial_scene_layout_service"]["service_name"]
        if not self.wait_for_ign_service(service_name):
            return ""
        try:
            output = subprocess.check_output(self.initial_scene_layout_cmd, universal_newlines=True)
            self.get_logger().info("Initial scene info retrieved.")
            return output
        except Exception as e:
            self.get_logger().error(f"Error calling ign service: {e}")
            return ""

    def timer_callback(self):
        """
        Periodic callback that reads pose updates from Gazebo and updates object poses.
        """
        try:
            output = subprocess.check_output(self.pose_update_cmd, universal_newlines=True)
        except Exception as e:
            self.get_logger().error(f"Error reading pose topic: {e}")

        pose_info = self.parse_pose_info(output)

        # Process each object using your rules.
        for obj in self.objects:
            if not self.match_interests(obj["name"]):
                continue
            #print(obj["name"] + ":")
            tokens = obj["name"].split("_")
            if tokens[-1].lower() == "visual":
                # Case 1: Object name ends with '_visual'
                search_name = "_".join(tokens[:-1])
                if search_name in pose_info:
                    obj["pose"] = self.merge_pose(pose_info[search_name], obj["link_pose"])
                    self.get_logger().info(f"Updated {obj['name']} using pose for '{search_name}'")
                else:
                    self.get_logger().warn(f"No pose found for '{search_name}'")
            else:
                # Case 2: Object name is composite (e.g., 'Table_back_left_leg')
                # Find the first token that is one of the objects_of_interest.
                first = None
                for token in tokens:
                    if token in self.objects_of_interest:
                        first = token
                        break
                if first is not None:
                    remaining_tokens = tokens[tokens.index(first)+1:]
                    second = "_".join(remaining_tokens) if remaining_tokens else ""
                    pose_first = pose_info.get(first, {})
                    pose_second = pose_info.get(second, {})
                    updated_pose = self.merge_pose(pose_first, pose_second)
                    obj["pose"] = self.merge_pose(updated_pose, obj["link_pose"])
                    self.get_logger().info(f"Updated {obj['name']} using merged pose from '{first}' and '{second}'")
                else:
                    if obj["name"] in pose_info:
                        obj["pose"] = self.merge_pose(pose_info[obj["name"]], obj["link_pose"])
                        self.get_logger().info(f"Updated {obj['name']} using its own pose")
                    else:
                        self.get_logger().warn(f"No matching pose found for {obj['name']}")
        
        # After updating all object poses, update the planning scene.
        self.update_planning_scene()


    def create_marker(self, model, action=Marker.ADD):
        """
        Convert *model* dict into a visualization Marker.
        If *action*==Marker.DELETE, only id/ns/action/header are populated.
        """
        marker            = Marker()
        marker.header.frame_id  = 'world'
        marker.header.stamp     = self.get_clock().now().to_msg()
        marker.ns        = 'scene_objects'          # keep one namespace
        marker.id        = hash(model['name']) & 0x7FFFFFFF   # stable int32
        marker.action    = action

        if action == Marker.DELETE:
            return marker     # nothing else required

        pose = model.get('pose', {})
        pos  = pose.get('position', {})
        ori  = pose.get('orientation', {})
        marker.pose.position.x = pos.get('x', 0.0)
        marker.pose.position.y = pos.get('y', 0.0)
        marker.pose.position.z = pos.get('z', 0.0)
        marker.pose.orientation.x = ori.get('x', 0.0)
        marker.pose.orientation.y = ori.get('y', 0.0)
        marker.pose.orientation.z = ori.get('z', 0.0)
        marker.pose.orientation.w = ori.get('w', 1.0)

        geom = model.get('geometry', {})
        gtyp = geom.get('type', 'box')
        if gtyp == 'box':
            marker.type  = Marker.CUBE
            marker.scale.x = geom.get('size', {}).get('x', 1.0)
            marker.scale.y = geom.get('size', {}).get('y', 1.0)
            marker.scale.z = geom.get('size', {}).get('z', 1.0)
        elif gtyp == 'cylinder':
            marker.type  = Marker.CYLINDER
            marker.scale.x = marker.scale.y = 2.0 * geom.get('radius', 0.05)
            marker.scale.z = geom.get('length', 0.1)
        elif gtyp == 'mesh':
            marker.type         = Marker.MESH_RESOURCE
            marker.mesh_resource = geom.get('filename', '')
            marker.mesh_use_embedded_materials = True
            s = geom.get('scale', 1.0)
            marker.scale.x = marker.scale.y = marker.scale.z = s
        else:         # default: 5 cm cube
            marker.type  = Marker.CUBE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05

        marker.color = self.rgba_from_material(model.get('material', {}))

        return marker


    def scene_add_markers(self, req, res):
        """
        Publish RViz markers (instead of collision objects) for every object
        that matches self.match_interests().
        """
        ma = MarkerArray()

        for obj in self.objects:
            if not self.match_interests(obj['name']):
                continue
            ma.markers.append(self.create_marker(obj, action=Marker.ADD))

        self.marker_pub.publish(ma)
        self.get_logger().info(f'Published {len(ma.markers)} markers.')

        res.success  = True
        res.message  = f'{len(ma.markers)} markers added to RViz'
        return res

    def scene_remove_markers(self, req, res):
        """
        Remove the previously-added markers from RViz.
        """
        ma = MarkerArray()
        for obj in self.objects:
            if not self.match_interests(obj['name']):
                continue
            ma.markers.append(self.create_marker(obj, action=Marker.DELETE))

        self.marker_pub.publish(ma)
        self.get_logger().info(f'Removed {len(ma.markers)} markers.')

        res.success  = True
        res.message  = f'{len(ma.markers)} markers removed'
        return res

    def scene_update_objects(self, req, res):
        self.get_logger().info(f"REQ  name='{req.name}'  action='{req.action}'")

        ps = PlanningScene()
        ps.is_diff = True

        if req.action == "keep_only":
            for obj in self.objects:
                if not self.match_interests(obj["name"]):
                    continue
                # as we keep the req.name only
                if req.name.lower() in obj["name"].lower():
                    obj_id = obj["name"]
                    continue

                co = obj.get("collision_object", None)
                if co is None:
                    continue  # Skip if no collision object stored

                remove = CollisionObject()
                remove.id        = co.id
                remove.operation = CollisionObject.REMOVE
                ps.world.collision_objects.append(remove)
            
            self.planning_scene_pub.publish(ps)
            self.get_logger().info(f"Only keep the '{obj_id}' in the planning scene.")
            self.scene_objects_deleted = True
            res.success = True
            res.message = f"Only keep the '{obj_id}' in the seen."

        return res

    def scene_remove_objects(self, req, res):
        """
        Updates the planning scene by removing the stored collision objects.
        """
        ps = PlanningScene()
        ps.is_diff = True

        for obj in self.objects:
            if not self.match_interests(obj["name"]):
                continue
            # Retrieve the pre-built collision object
            co = obj.get("collision_object", None)
            if co is None:
                continue  # Skip if no collision object stored

            remove = CollisionObject()
            remove.id        = co.id
            remove.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(remove)
            
        #self.planning_scene_pub.publish(rs)
        self.planning_scene_pub.publish(ps)
        self.get_logger().info("Published removed collision objects to the planning scene.")
        
        self.scene_objects_deleted = True
        # After updating all object poses, update the planning scene.
        res.success = True
        res.message = f'scene collision objects removed'

        return res

    def scene_add_objects(self, req, res):
        """
        Updates the planning scene by adding the stored collision objects.
        """
        ps = PlanningScene()
        ps.is_diff = True

        for obj in self.objects:
            if not self.match_interests(obj["name"]):
                continue

            co_obj, color = self.create_collision_object(obj)
            ps.world.collision_objects.append(co_obj)
            ps.object_colors.append(color)
            
        #self.planning_scene_pub.publish(rs)
        self.planning_scene_pub.publish(ps)
        self.get_logger().info("Published removed collision objects to the planning scene.")
        
        # After updating all object poses, update the planning scene.
        res.success = True
        res.message = f'scene collision objects removed'

        return res

    def scene_update(self, req, res):
        """
        Periodic callback that reads pose updates from Gazebo and updates object poses.
        """
        try:
            output = subprocess.check_output(self.pose_update_cmd, universal_newlines=True)
        except Exception as e:
            self.get_logger().error(f"Error reading pose topic: {e}")
            res.success = False
            res.message = f"Error reading pose topic: {e}"
            return res

        print("===================SCENE_UPDATE====================")
        pose_info = self.parse_pose_info(output)

        # Process each object using your rules.
        for obj in self.objects:
            if not self.match_interests(obj["name"]):
                continue
            #print(obj["name"] + ":")
            tokens = obj["name"].split("_")
            if tokens[-1].lower() == "visual":
                # Case 1: Object name ends with '_visual'
                search_name = "_".join(tokens[:-1])
                if search_name in pose_info:
                    obj["pose"] = self.merge_pose(pose_info[search_name], obj["link_pose"])
                    self.get_logger().info(f"Updated {obj['name']} using pose for '{search_name}'")
                else:
                    self.get_logger().warn(f"No pose found for '{search_name}'")
            else:
                # Case 2: Object name is composite (e.g., 'Table_back_left_leg')
                # Find the first token that is one of the objects_of_interest.
                first = None
                for token in tokens:
                    if token in self.objects_of_interest:
                        first = token
                        break
                if first is not None:
                    remaining_tokens = tokens[tokens.index(first)+1:]
                    second = "_".join(remaining_tokens) if remaining_tokens else ""
                    pose_first = pose_info.get(first, {})
                    pose_second = pose_info.get(second, {})
                    updated_pose = self.merge_pose(pose_first, pose_second)
                    obj["pose"] = self.merge_pose(updated_pose, obj["link_pose"])
                    self.get_logger().info(f"Updated {obj['name']} using merged pose from '{first}' and '{second}'")
                else:
                    if obj["name"] in pose_info:
                        obj["pose"] = self.merge_pose(pose_info[obj["name"]], obj["link_pose"])
                        self.get_logger().info(f"Updated {obj['name']} using its own pose")
                    else:
                        self.get_logger().warn(f"No matching pose found for {obj['name']}")
        
        # After updating all object poses, update the planning scene.
        self.update_planning_scene()
        res.success = True
        res.message = f'scene updated'

        return res

    def update_planning_scene(self):
        """
        Updates the planning scene by modifying the stored collision objects.
        For each object, the stored collision object's pose is updated and its operation
        is set to MOVE before publishing the diff.
        """
        ps = PlanningScene()
        ps.is_diff = True

        #rs = PlanningScene()
        #rs.is_diff = True
        for obj in self.objects:
            if not self.match_interests(obj["name"]):
                continue
            # Retrieve the pre-built collision object
            co = obj.get("collision_object", None)
            if co is None and not self.scene_objects_deleted:
                continue  # Skip if no collision object stored
            
            if not self.scene_objects_deleted: 
                co_new = CollisionObject()

                # Update the pose from the current object pose (updated from the ign topic)
                pose = Pose()
                pos = obj["pose"].get("position", {})
                ori = obj["pose"].get("orientation", {})

                pose.position.x = round(pos.get("x", 0.0), 3)
                pose.position.y = round(pos.get("y", 0.0), 3)
                pose.position.z = self.update_height(round(pos.get("z", 0.0), 3), 0.005)

                pose.orientation.x = round(ori.get("x", 0.0), 3)
                pose.orientation.y = round(ori.get("y", 0.0), 3)
                pose.orientation.z = round(ori.get("z", 0.0), 3)
                pose.orientation.w = round(ori.get("w", 1.0), 3)

                print("Pose for ", obj["name"])
                print(pose)

                #print(obj["name"], pose)
                co_new.header = co.header
                co_new.id = co.id
                co_new.pose = pose
                co_new.operation = CollisionObject.MOVE
                ps.world.collision_objects.append(co_new)

            else:
                co_obj, color = self.create_collision_object(obj)
                ps.world.collision_objects.append(co_obj)
                ps.object_colors.append(color)

                '''
                # Update the stored collision object's pose.
                # We assume the collision object has either a primitive or a mesh.
                if co.primitives:
                    # For primitive geometries.
                    if len(co.primitive_poses) > 0:
                        #print("Update Primitive " + obj["name"] + ":", co.primitives[0].type, pose)
                        co.primitive_poses[0] = pose
                    else:
                        co.primitive_poses.append(pose)
                elif co.meshes:
                    # For mesh geometries.
                    if len(co.mesh_poses) > 0:
                        #print("Update Mesh " + obj["name"] + ":", pose)
                        co.mesh_poses[0] = pose
                    else:
                        co.mesh_poses.append(pose)
                else:
                    # If neither exists, you could store the pose in a generic field if needed.
                    pass

                # Set the operation to MOVE (or MODIFY, depending on your interface)
                co.operation = CollisionObject.ADD  # Alternatively, use CollisionObject.MODIFY if appropriate.
            
                # Add the updated collision object to the planning scene diff.
                ps.world.collision_objects.append(co)
                '''

        #self.planning_scene_pub.publish(rs)
        self.scene_objects_deleted = False
        self.planning_scene_pub.publish(ps)
        self.get_logger().info("Published updated collision object poses to the planning scene.")

    def match_interests(self, string):
        tokens = string.split('_')

        if tokens[-1] == 'visual':  
            substring = '_'.join(tokens[:-1])  
        else:  
            substring = tokens[0]  

        if substring in self.objects_of_interest:
            return True
        else:
            return False

    def extract_block(self, text, start_keyword):
        """
        Extract a block of text starting at start_keyword and ending at the matching closing brace.
        Returns the substring including start_keyword.
        """
        index = text.find(start_keyword)
        if index == -1:
            return None
        # Find the first '{' after the keyword.
        open_brace_index = text.find("{", index)
        if open_brace_index == -1:
            return None

        count = 0
        i = open_brace_index
        while i < len(text):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    return text[index:i+1]
            i += 1
        return None

    def extract_all_blocks(self, text, start_keyword):
        """
        Extract all blocks in text that start with start_keyword.
        Returns a list of block strings.
        """
        blocks = []
        start = 0
        while True:
            index = text.find(start_keyword, start)
            if index == -1:
                break
            block = self.extract_block(text[index:], start_keyword)
            if block:
                blocks.append(block)
                start = index + len(block)
            else:
                break
        return blocks

    def extract_pose_fields(self, block, tag="pose {"):
        """
        Extracts position and orientation values from a pose block.
        Returns a dictionary that may contain:
           { "position": {"x":..., "y":..., "z":...},
             "orientation": {"x":..., "y":..., "z":..., "w":...} }
        Only fields that are explicitly specified are returned.
        """
        pose_block = self.extract_block(block, tag)
        result = {}
        if not pose_block:
            return result

        pos_block = self.extract_block(pose_block, "position {")
        pos = {}
        if pos_block:
            for axis in ['x', 'y', 'z']:
                m = re.search(rf'{axis}:\s*([-\d\.e]+)', pos_block)
                if m:
                    pos[axis] = float(m.group(1))
                else:
                    pos[axis] = 0.0
        result['position'] = pos

        ori_block = self.extract_block(pose_block, "orientation {")
        ori = {}
        if ori_block:
            for comp in ['x', 'y', 'z', 'w']:
                m = re.search(rf'{comp}:\s*([-\d\.e]+)', ori_block)
                if m:
                    ori[comp] = float(m.group(1))
                else:
                    ori[comp] = 0.0
        result['orientation'] = ori

        return result

    def parse_pose_info(self, text):
        """
        Parses the output from 'ign topic ...' into a dictionary mapping pose name to pose data.
        """
        pose_blocks = self.extract_all_blocks(text, "pose {")
        pose_dict = {}
        for block in pose_blocks:
            m = re.search(r'name:\s*"([^"]+)"', block)
            if not m:
                continue
            name = m.group(1)
            pose_data = self.extract_pose_fields(block, "pose {")
            pose_dict[name] = pose_data
        return pose_dict

    def merge_pose(self, p1, p2):
        """Compose two poses in XYZ+quaternion form (q_parent ⊕ q_child)."""

        # ---------------- parent pose as homogeneous matrix -----------------
        q1 = [
            p1['orientation']['x'],
            p1['orientation']['y'],
            p1['orientation']['z'],
            p1['orientation']['w'],
        ]
        t1 = [
            p1['position']['x'],
            p1['position']['y'],
            p1['position']['z'],
        ]
        T1 = tft.quaternion_matrix(q1)
        T1[0:3, 3] = t1

        # ---------------- pose2 as homogeneous matrix ------------------
        q2 = [
            p2['orientation']['x'],
            p2['orientation']['y'],
            p2['orientation']['z'],
            p2['orientation']['w'],
        ]
        t2 = [
            p2['position']['x'],
            p2['position']['y'],
            p2['position']['z'],
        ]
        T2 = tft.quaternion_matrix(q2)
        T2[0:3, 3] = t2

        T = T1 @ T2                                         # matrix multiplication
        q = tft.quaternion_from_matrix(T)

        return {
            'position': {'x': T[0, 3], 'y': T[1, 3], 'z': T[2, 3]},
            'orientation': {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]},
        }

    def parse_scene(self, text):
        """
        Parse an Ignition scene description text and return a list of objects.
        Each object corresponds to a visual tag, with a merged pose and geometry.
        This function supports multiple model tags.
        """
        objects = []
        # Extract all model blocks
        model_blocks = self.extract_all_blocks(text, "model {")
        for model_block in model_blocks:
            # Get the model name.
            name_match = re.search(r'name:\s*"([^"]+)"', model_block)
            if not name_match:
                continue
            model_name = name_match.group(1)
            # Extract parent pose from the model block.
            model_pose = self.extract_pose_fields(model_block, "pose {")

            link_block = self.extract_all_blocks(model_block, "link {")
            link_pose = self.extract_pose_fields(link_block[0], "pose {")
            #print(model_name, ", link_pose", link_pose)
            world_pose = self.merge_pose(model_pose, link_pose)

            # Extract all visual blocks from the model block.
            visual_blocks = self.extract_all_blocks(model_block, "visual {")
            if not visual_blocks:
                # If no visual blocks, use the model as one object.
                obj = {"name": model_name, "pose": world_pose}
                objects.append(obj)
            else:
                for visual_block in visual_blocks:
                    # Get visual name if available.
                    visual_name_match = re.search(r'name:\s*"([^"]+)"', visual_block)
                    visual_name = visual_name_match.group(1) if visual_name_match else ""
                    # Extract visual's own pose (if provided).
                    visual_pose = self.extract_pose_fields(visual_block, "pose {")
                    # Merge parent's pose with visual's pose (visual overrides parent's values)
                    world_pose = self.merge_pose(world_pose, visual_pose)
                    
                    # Extract geometry from the visual block.
                    geometry = None
                    geometry_block = self.extract_block(visual_block, "geometry {")
                    if geometry_block:
                        # Check for box geometry: support "type: box" or "box {"
                        if re.search(r'(type:\s*box)|(box\s*{)', geometry_block, re.IGNORECASE):
                            if re.search(r'box\s*{', geometry_block, re.IGNORECASE):
                                box_block = self.extract_block(geometry_block, "box {")
                                size_block = self.extract_block(box_block, "size {")
                            else:
                                size_block = self.extract_block(geometry_block, "size {")
                            dimensions = {}
                            if size_block:
                                for axis in ['x','y','z']:
                                    m = re.search(rf'{axis}:\s*([-\d\.e]+)', size_block)
                                    dimensions[axis] = float(m.group(1)) if m else 1.0
                            geometry = {'type': 'box', 'size': dimensions}
                        # Check for cylinder geometry
                        elif re.search(r'(type:\s*cylinder)|(cylinder\s*{)', geometry_block, re.IGNORECASE):
                            if re.search(r'cylinder\s*{', geometry_block, re.IGNORECASE):
                                cyl_block = self.extract_block(geometry_block, "cylinder {")
                                radius_match = re.search(r'radius:\s*([-\d\.e]+)', cyl_block)
                                length_match = re.search(r'length:\s*([-\d\.e]+)', cyl_block)
                            else:
                                radius_match = re.search(r'radius:\s*([-\d\.e]+)', geometry_block)
                                length_match = re.search(r'length:\s*([-\d\.e]+)', geometry_block)
                            if radius_match and length_match:
                                geometry = {
                                    'type': 'cylinder',
                                    'radius': float(radius_match.group(1)),
                                    'length': float(length_match.group(1))
                                }
                        # Check for mesh geometry
                        elif re.search(r'(type:\s*mesh)|(mesh\s*{)', geometry_block, re.IGNORECASE):
                            if re.search(r'mesh\s*{', geometry_block, re.IGNORECASE):
                                mesh_block = self.extract_block(geometry_block, "mesh {")
                                filename_match = re.search(r'filename:\s*"([^"]+)"', mesh_block)
                            else:
                                filename_match = re.search(r'filename:\s*"([^"]+)"', geometry_block)
                            if filename_match:
                                geometry = {'type': 'mesh', 'filename': filename_match.group(1)}
                    
                    material = None
                    material_block = self.extract_block(visual_block, "material {")
                    if material_block:
                        if re.search(r'ambient\s*{', material_block, re.IGNORECASE):
                            ambient_block = self.extract_block(material_block, "ambient {")
                        ambient_rgba = {}
                        if ambient_block:
                            for element in ['r','g','b', 'a']:
                                m = re.search(rf'{element}:\s*([-\d\.e]+)', ambient_block)
                                ambient_rgba[element] = float(m.group(1)) if m else 0.0
                        if re.search(r'diffuse\s*{', material_block, re.IGNORECASE):
                            diffuse_block = self.extract_block(material_block, "diffuse {")
                        diffuse_rgba = {}
                        if diffuse_block:
                            for element in ['r','g','b', 'a']:
                                m = re.search(rf'{element}:\s*([-\d\.e]+)', diffuse_block)
                                diffuse_rgba[element] = float(m.group(1)) if m else 0.0
                        material = {'ambient': ambient_rgba, 'diffuse': diffuse_rgba}

                    # Compose the object name by combining model and visual names.
                    obj_name = f"{model_name}_{visual_name}" if visual_name else model_name
                    obj = {
                        "name": obj_name,
                        "pose": world_pose,
                        "link_pose": link_pose
                    }
                    if geometry:
                        obj["geometry"] = geometry
                    if material:
                        obj["material"] = material 
                    objects.append(obj)
        return objects

    def rgba_from_material(self, material):
        """
        Returns {'r':..,'g':..,'b':..,'a':..} from a material block.
        Priority: diffuse -> ambient -> default(0.8,0.8,0.8,1)
        """
        rgba_src = material.get('diffuse') or material.get('ambient') or {}
        return ColorRGBA(
            r = rgba_src.get('r', 0.8),
            g = rgba_src.get('g', 0.8),
            b = rgba_src.get('b', 0.8),
            a = rgba_src.get('a', 1.0)
        )

    def update_height(self, h, offset):
        level = int(round((h - self.z_base) / self.z_inc))
        z_updated = self.z_base + level * self.z_inc + level * offset
        z_updated = round(z_updated, 3)

        return z_updated

    def create_collision_object(self, model):
        """
        Converts a model dictionary into a MoveIt2 CollisionObject.
        """
        collision_object = CollisionObject()
        collision_object.id = model.get('name')
        collision_object.header.frame_id = "world"
        collision_object.operation = CollisionObject.ADD

        # Create the pose for the object.
        pose = Pose()
        pos = model.get('pose', {}).get('position', {})
        ori = model.get('pose', {}).get('orientation', {})
        pose.position.x = pos.get('x', 0.0)
        pose.position.y = pos.get('y', 0.0)
        pose.position.z = self.update_height(pos.get('z', 0.0), 0.005)
        pose.orientation.x = ori.get('x', 0.0)
        pose.orientation.y = ori.get('y', 0.0)
        pose.orientation.z = ori.get('z', 0.0)
        pose.orientation.w = ori.get('w', 1.0)

        # Depending on the geometry type, create either a primitive or a mesh.
        if 'geometry' in model:
            geom = model['geometry']
            if geom['type'] == 'box':
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                dims = geom.get('size', {})
                primitive.dimensions = [
                    dims.get('x', 1.0),
                    dims.get('y', 1.0),
                    dims.get('z', 1.0)
                ]
                collision_object.primitives.append(primitive)
                collision_object.primitive_poses.append(pose)
            elif geom['type'] == 'cylinder':
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.CYLINDER
                # The dimensions for a cylinder are [height, radius].
                collision_object.primitives.append(primitive)
                primitive.dimensions = [
                    geom.get('length', 1.0),
                    geom.get('radius', 0.1)
                ]
                collision_object.primitive_poses.append(pose)
            elif geom['type'] == 'mesh':
                # For meshes, you would normally load the mesh file into a shape_msgs/Mesh.
                # This example uses a placeholder function.
                mesh = self.load_mesh(geom['filename'])
                if mesh is not None:
                    collision_object.meshes.append(mesh)
                    collision_object.mesh_poses.append(pose)
                else:
                    self.get_logger().warn(f"Failed to load mesh for {model.get('name')}")

        model["collision_object"] = collision_object

        obj_color = ObjectColor()
        obj_color.id = collision_object.id
        obj_color.color = self.rgba_from_material(model.get('material', {}))

        model["collision_object_color"] = obj_color.color

        return collision_object, obj_color

    def load_mesh(self, filename):
        """
        Load a mesh from a given resource filename and convert it into a shape_msgs/Mesh.

        This function mimics the C++ approach using shapes::createMeshFromResource and
        shapes::constructMsgFromShape. It uses trimesh to load the mesh and then constructs
        a Mesh message that can be used in a CollisionObject.
    
        :param filename: A string representing the resource path, e.g.,
                     "package://sim_move/meshes/ring.STL"
        :return: A shape_msgs/Mesh message, or None if loading fails.
        """
        try:
            # Resolve package:// URI if necessary
            if filename.startswith("package://"):
                parts = filename[len("package://"):].split('/', 1)
                package_name = parts[0]
                relative_path = parts[1]
                package_share = get_package_share_directory(package_name)
                filepath = os.path.join(package_share, relative_path)
            elif filename.startswith("model://"):
                #parts = filename[len("model://"):].split('/', 1)
                parts = filename[len("model://"):]
                relative_path = parts[1]
                filepath = os.path.join("/home/hao/workspace/smash/models", parts)
                print(filepath)
            else:
                filepath = filename

            # Load the mesh using trimesh
            mesh = trimesh.load(filepath, force='mesh')
            if mesh is None:
                self.get_logger().error(f"Failed to load mesh from {filepath}")
                return None

            # Create a Mesh message and populate its vertices and triangles
            mesh_msg = Mesh()

            # Fill vertices (each vertex is a geometry_msgs/Point)
            for vertex in mesh.vertices:
                pt = Point(x=float(vertex[0]), y=float(vertex[1]), z=float(vertex[2]))
                mesh_msg.vertices.append(pt)

            # Fill triangles (each face is a MeshTriangle with vertex indices)
            for face in mesh.faces:
                triangle = MeshTriangle(vertex_indices=[int(face[0]), int(face[1]), int(face[2])])
                mesh_msg.triangles.append(triangle)

            self.get_logger().info(f"Mesh loaded from: {filepath}")
            return mesh_msg

        except Exception as e:
            self.get_logger().error("Exception in load_mesh: " + str(e))
            return None

def main(args=None):
    parser = argparse.ArgumentParser(description="Gazebo Pose Updater.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--ready-flag", default="")

    args, unknown = parser.parse_known_args()

    ready_flag = pathlib.Path(args.ready_flag) if args.ready_flag else None

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    rclpy.init(args=None)
    node = GazeboPoseUpdater(config, ready_flag)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

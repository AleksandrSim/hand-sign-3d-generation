import numpy as np

class JointDataConverter:
    def __init__(self):
        self.joint_data = None  # Placeholder, not used in this test
    def calculate_euler_angles(self, parent_pos, joint_pos, child_pos):
        # Ensure correct shaping for the operations
        parent_pos = np.array(parent_pos).reshape((1, 3))
        joint_pos = np.array(joint_pos).reshape((1, 3))
        child_pos = np.array(child_pos).reshape((1, 3))

        # Calculate vectors
        bone_vector = joint_pos - parent_pos
        next_bone_vector = child_pos - joint_pos
        
        # Normalize the vectors
        bone_vector_norm = bone_vector / np.linalg.norm(bone_vector, axis=1, keepdims=True)
        next_bone_vector_norm = next_bone_vector / np.linalg.norm(next_bone_vector, axis=1, keepdims=True)
        
        # Calculate yaw (Z-axis rotation)
        yaw = np.arctan2(bone_vector_norm[0, 1], bone_vector_norm[0, 0])
        
        # Pitch (Y-axis rotation)
        pitch = np.arcsin(-bone_vector_norm[0, 2])
        
        # Roll (X-axis rotation) - For simplicity, assuming roll is 0 in this scenario
        roll = 0.0

        # Convert radians to degrees
        euler_angles_deg = np.array([np.rad2deg(pitch), np.rad2deg(roll), np.rad2deg(yaw)])

        return euler_angles_deg
    
    def test_euler_angles(self):
        # Define test vectors
        parent_pos = np.array([[[0, 0, 0]]])  # Origin
        joint_pos = np.array([[[1, 0, 0]]])  # Along the X-axis
        child_pos = np.array([[[1, 1, 0]]])  # Move along the Y-axis from the joint

        # Calculate Euler angles for the test vectors
        euler_angles_test = self.calculate_euler_angles(parent_pos, joint_pos, child_pos)
        print("Test Euler Angles (Pitch, Roll, Yaw) in Degrees:", euler_angles_test)






# Using the tester
#converter = JointDataConverter()
#converter.test_euler_angles()

parent_pos = np.array([0, 0, 0])
joint_pos = np.array([1, 0, 0])
child_pos = np.array([1, 1, 0])

# Calculate the direction vectors
bone_vector = joint_pos - parent_pos
next_bone_vector = child_pos - joint_pos

# Manually calculate yaw
yaw = np.arctan2(next_bone_vector[1], next_bone_vector[0])
print("Yaw in radians:", yaw)
print("Yaw in degrees:", np.rad2deg(yaw))

pitch = np.arctan2(-bone_vector[2], np.sqrt(bone_vector[0]**2 + bone_vector[1]**2))
pitch_deg = np.rad2deg(pitch)
print("Pitch in degrees:", pitch_deg)

# Roll calculation (simplifies to 0 for this scenario)
roll = 0.0  # Given no Z component change or twist around the bone vector
roll_deg = np.rad2deg(roll)
print("Roll in degrees:", roll_deg)



print('Expected Yaw 90 degrees else 0, 0 ')
parent_pos = np.array([0, 0, 0])
joint_pos = np.array([1, 0, 1])  # Includes upward movement along the Z-axis
child_pos = np.array([1, 1, 1])  # Maintains Z-height from the joint position

# Calculate the direction vectors
bone_vector = joint_pos - parent_pos
next_bone_vector = child_pos - joint_pos

# Yaw calculation (should remain unaffected by the vertical component in the bone vector)
yaw = np.arctan2(next_bone_vector[1], next_bone_vector[0])
yaw_deg = np.rad2deg(yaw)
print("Yaw in degrees:", yaw_deg)

# Pitch calculation (now should reflect the vertical movement from the parent to the joint)
# Using the bone vector's Z component to calculate the tilt up or down
pitch = np.arctan2(bone_vector[2], np.sqrt(bone_vector[0]**2 + bone_vector[1]**2))
pitch_deg = np.rad2deg(pitch)
print("Pitch in degrees:", pitch_deg)

# Roll calculation (still simplifies to 0 as there's no twist around the bone vector in this scenario)
roll = 0.0  # Given the scenario setup, roll is expected to be 0
roll_deg = np.rad2deg(roll)
print("Roll in degrees:", roll_deg)




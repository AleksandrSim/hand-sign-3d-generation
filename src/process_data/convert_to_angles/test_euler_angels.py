import numpy as np

def calculate_euler_angles(bone_vector, next_bone_vector, initial_up=np.array([0, 0, 1]), final_up=np.array([0, 1, 0])):
    # Yaw calculation: Angle in the XY plane from the X-axis
    yaw = np.arctan2(next_bone_vector[1], next_bone_vector[0])
    
    # Pitch calculation: Tilt up or down from the XY plane
    pitch = np.arctan2(-bone_vector[2], np.sqrt(bone_vector[0]**2 + bone_vector[1]**2))
    
    # Roll calculation: Assuming change in 'up' vector orientation due to rotation around the bone vector
    cos_angle = np.dot(initial_up, final_up) / (np.linalg.norm(initial_up) * np.linalg.norm(final_up))
    roll = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to ensure value is within domain of arccos
    
    return np.rad2deg(pitch), np.rad2deg(roll), np.rad2deg(yaw)

def unified_test_euler_angles(parent_pos, joint_pos, child_pos):
    # Unified scenario incorporating movements affecting yaw, pitch, and roll# Diagonal movement affecting yaw and simulating pitch/roll change
    
    # Calculate the direction vectors
    bone_vector = joint_pos - parent_pos
    next_bone_vector = child_pos - joint_pos
    
    # Simulated 'initial_up' and 'final_up' vectors for roll calculation
    initial_up = np.array([0, 0, 1])  # Starting 'up' vector orientation
    final_up = np.array([0, 1, 0])    # Simulated 'up' vector orientation after roll
    
    pitch_deg, roll_deg, yaw_deg = calculate_euler_angles(bone_vector, next_bone_vector, initial_up, final_up)
    print(f"Unified Test - Yaw: {yaw_deg} degrees, Pitch: {pitch_deg} degrees, Roll: {roll_deg} degrees")


def test_no_movement():
    parent_pos = np.array([0, 0, 0])
    joint_pos = parent_pos  # No movement
    child_pos = joint_pos  # No movement
    print("\nTest Case 1: No Movement")
    unified_test_euler_angles(parent_pos, joint_pos, child_pos)

def test_pure_pitch():
    parent_pos = np.array([0, 0, 0])
    joint_pos = np.array([0, 0, 1])  # Upwards along the Z-axis
    child_pos = np.array([0, 0, 2])  # Further up along the Z-axis
    print("\nTest Case 2: Pure Pitch Movement")
    unified_test_euler_angles(parent_pos, joint_pos, child_pos)

# Original Unified Test for Reference
print("Original Unified Test:")
unified_test_euler_angles(parent_pos = np.array([0, 0, 0]),
                            joint_pos = np.array([1, 0, 0]),
                            child_pos = np.array([1, 1, 1]))

# Execute additional test functions
test_no_movement()
test_pure_pitch()



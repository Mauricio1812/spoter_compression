import numpy as np
import torch


##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    valid_sequence = True
    last_starting_point, last_ending_point = None, None

    # Prevent from even starting the analysis if some necessary elements are not present
    if (data[body_dict['pose_left_shoulder']][0] == 0.0 or data[body_dict['pose_right_shoulder']][0] == 0.0):
        if not last_starting_point:
            valid_sequence = False
        else:
            starting_point, ending_point = last_starting_point, last_ending_point
    else:
        # NOTE:
        #
        # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
        # this is meant for the distance between the very ends of one's shoulder, as literature studying body
        # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
        # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
        # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
        #
        # Please, review this if using other third-party pose estimation libraries.

        if data[body_dict['pose_left_shoulder']][0] != 0 and data[body_dict['pose_right_shoulder']][0] != 0:
            
            left_shoulder = data[body_dict['pose_left_shoulder']]
            right_shoulder = data[body_dict['pose_right_shoulder']]

            shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                    (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

            mid_distance = (0.5,0.5)#(left_shoulder - right_shoulder)/2
            head_metric = shoulder_distance/2
        # Set the starting and ending point of the normalization bounding box
        starting_point = [mid_distance[0] - 3 * head_metric, data[body_dict['pose_right_eye']][1] - (head_metric / 2)]
        ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 4.5 * head_metric]

        last_starting_point, last_ending_point = starting_point, ending_point

    # Normalize individual landmarks and save the results
    for pos, kp in enumerate(data):
        
        # Prevent from trying to normalize incorrectly captured points
        if data[pos][0] == 0:
            continue

        normalized_x = (data[pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                starting_point[0])
        normalized_y = (data[pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                ending_point[1])

        data[pos][0] = normalized_x
        data[pos][1] = 1 - normalized_y
            
    return data
################################################
# Function that normalize the hands (but also the face)
################################################
def normalize_hand(data, body_section_dict):
    # Retrieve all of the X and Y values of the current frame
    landmarks_x_values = data[:, 0]
    landmarks_y_values = data[:, 1]

    # Calculate the deltas
    width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
        landmarks_y_values)
    if width > height:
        delta_x = 0.1 * width
        delta_y = delta_x + ((width - height) / 2)
    else:
        delta_y = 0.1 * height
        delta_x = delta_y + ((height - width) / 2)

    # Set the starting and ending point of the normalization bounding box
    starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
    ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

    # Normalize individual landmarks and save the results
    for pos, kp in enumerate(data):
        # Prevent from trying to normalize incorrectly captured points
        if data[pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                starting_point[1] - ending_point[1]) == 0:
            continue

        normalized_x = (data[pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                starting_point[0])
        normalized_y = (data[pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                starting_point[1])

        data[pos][0] = normalized_x
        data[pos][1] = normalized_y

    return data


def normalize_pose_hands_function(data, body_section, body_part):

    pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
    face = [pos for pos, body in enumerate(body_section) if body == 'face']
    leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
    rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']

    body_section_dict = {body:pos for pos, body in enumerate(body_part)}

    assert len(pose) > 0 and len(leftHand) > 0 and len(rightHand) > 0 #and len(face) > 0

    for index_video in range(len(data)):
        data[index_video][pose,:] = normalize_pose(data[index_video][pose,:], body_section_dict)
        #data[index_video][face,:] = normalize_hand(data[index_video][face,:], body_section_dict)
        data[index_video][leftHand,:] = normalize_hand(data[index_video][leftHand,:], body_section_dict)
        data[index_video][rightHand,:] = normalize_hand(data[index_video][rightHand,:], body_section_dict)

    return data
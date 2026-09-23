CONFIG = {
    "name": 'Squat from side with  spine curvative estimation',
    "camera_view": 'side',
    "client_instructions": 'Stand with either side towards the camera and perform squats. Aim for a 60 degree knee bend angle. Maintain a natural form of your back, avoid exessive rounding of the upper back  and too much of an arch in the lower back',
    "display_values": '- knee flexion angle\n- upper body lean angle\n- hip flexion angle\n- Thoracic (upper back) spine bend, in degrees\n- Lumbar (lower back) spine bend, in degrees',
    "session_duration_secs": 10,
    "feedback_mode": ["after_rep", "after_exercise"],
    "pose_backend": 'yolo11',
    "yolo_model": "yolo11n-pose.pt",
}


# List of all controls we have
ALL_CTRL = [
    "Thumb 01 R Ctrl", "Thumb 02 R Ctrl", "Thumb 03 R Ctrl",
    "Index Metacarpal R Ctrl", "Index 01 R Ctrl", "Index 02 R Ctrl",
    "Index 03 R Ctrl",
    "Middle Metacarpal R Ctrl", "Middle 01 R Ctrl", "Middle 02 R Ctrl",
    "Middle 03 R Ctrl",
    "Ring Metacarpal R Ctrl", "Ring 01 R Ctrl", "Ring 02 R Ctrl",
    "Ring 03 R Ctrl",
    "Pinky Metacarpal R Ctrl", "Pinky 01 R Ctrl", "Pinky 02 R Ctrl",
    "Pinky 03 R Ctrl",
    "Wrist R Ctrl", "Lowerarm R Ctrl", "Upperarm R Ctrl"
]

# Neutral pose for the hand and the arm
NEUTRAL_POSE: dict[str, tuple[float, float, float]] = {
    'Thumb 01 R Ctrl': (0.0, -20.0, 10.0),
    'Index 01 R Ctrl': (0.0, 0.0, -14.0),
    'Index 02 R Ctrl': (0.0, 7.0, -14.0),
    'Index 03 R Ctrl': (0.0, 0.0, -3.5),
    'Middle 01 R Ctrl': (0.0, 0.0, -12.0),
    'Middle 02 R Ctrl': (0.0, 7.0, -16.0),
    'Middle 03 R Ctrl': (0.0, 0.0, -5.0),
    'Ring 01 R Ctrl': (0.0, 0.0, -14.0),
    'Ring 02 R Ctrl': (0.0, 0.0, -24.0),
    'Ring 03 R Ctrl': (0.0, 0.0, -14.0),
    'Pinky 01 R Ctrl': (0.0, -8.0, -12.0),
    'Pinky 02 R Ctrl': (0.0, 0.0, -20.0),
    'Pinky 03 R Ctrl': (0.0, 0.0, -8.0),
    'Wrist R Ctrl': (0.0, 0.0, 0.0),
    'Lowerarm R Ctrl': (0.0, 0.0, 65.0),
    'Upperarm R Ctrl': (30.0, 0.0, 0.0)
}


# Define how to calculate angles for the controls
CONTROLS = {
    # Thumb 01 R Ctrl has a different logic
    'Thumb 02 R Ctrl': ('RightFinger1Metacarpal', 'RightFinger1Proximal',
                        'RightFinger1Distal'),
    'Thumb 03 R Ctrl': ('Thumb 02 R Ctrl', ),
    'Index 01 R Ctrl': ('RightFinger2Metacarpal', 'RightFinger2Proximal',
                        'RightFinger2Medial'),
    'Index 02 R Ctrl': ('RightFinger2Proximal', 'RightFinger2Medial',
                        'RightFinger2Distal'),
    'Index 03 R Ctrl': ('Index 02 R Ctrl', ),
    'Middle 01 R Ctrl': ('RightFinger3Metacarpal', 'RightFinger3Proximal',
                         'RightFinger3Medial'),
    'Middle 02 R Ctrl': ('RightFinger3Proximal', 'RightFinger3Medial',
                         'RightFinger3Distal'),
    'Middle 03 R Ctrl': ('Middle 02 R Ctrl', ),
    'Ring 01 R Ctrl': ('RightFinger4Metacarpal', 'RightFinger4Proximal',
                       'RightFinger4Medial'),
    'Ring 02 R Ctrl': ('RightFinger4Proximal', 'RightFinger4Medial',
                       'RightFinger4Distal'),
    'Ring 03 R Ctrl': ('Ring 02 R Ctrl', ),
    'Pinky 01 R Ctrl': ('RightFinger5Metacarpal', 'RightFinger5Proximal',
                        'RightFinger5Medial'),
    'Pinky 02 R Ctrl': ('RightFinger5Proximal', 'RightFinger5Medial',
                        'RightFinger5Distal'),
    'Pinky 03 R Ctrl': ('Pinky 02 R Ctrl', ),
}

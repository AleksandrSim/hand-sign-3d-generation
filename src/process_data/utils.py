import numpy as np


import numpy as np


HAND_BONES_CONNECTIONS = [
    ('RightHand', 'RightFinger5Metacarpal'),
    ('RightHand', 'RightFinger4Metacarpal'),
    ('RightHand', 'RightFinger3Metacarpal'),
    ('RightHand', 'RightFinger2Metacarpal'),
    ('RightHand', 'RightFinger1Metacarpal'),

    ('RightFinger1Metacarpal', 'RightFinger1Proximal'),
    ('RightFinger1Proximal', 'RightFinger1Distal'),
#    ('RightFinger1Distal', 'RightFinger1Tip'),
    
    ('RightFinger2Metacarpal', 'RightFinger2Proximal'),
    ('RightFinger2Proximal', 'RightFinger2Medial'),
    ('RightFinger2Medial', 'RightFinger2Distal'),
#    ('RightFinger2Distal', 'RightFinger2Tip'),
    
    ('RightFinger3Metacarpal', 'RightFinger3Proximal'),
    ('RightFinger3Proximal', 'RightFinger3Medial'),
    ('RightFinger3Medial', 'RightFinger3Distal'),
#    ('RightFinger3Distal', 'RightFinger3Tip'),
    
    ('RightFinger4Metacarpal', 'RightFinger4Proximal'),
    ('RightFinger4Proximal', 'RightFinger4Medial'),
    ('RightFinger4Medial', 'RightFinger4Distal'),
#    ('RightFinger4Distal', 'RightFinger4Tip'),
    
    ('RightFinger5Metacarpal', 'RightFinger5Proximal'),
    ('RightFinger5Proximal', 'RightFinger5Medial'),
    ('RightFinger5Medial', 'RightFinger5Distal'),
#    ('RightFinger5Distal', 'RightFinger5Tip')
]

HAND_BONES = ['RightHand',
              'RightFinger1Metacarpal', 'RightFinger1Proximal', 'RightFinger1Distal', 'RightFinger5Metacarpal',
                'RightFinger5Proximal', 'RightFinger5Medial', 'RightFinger5Distal', 'RightFinger4Metacarpal',
                  'RightFinger4Proximal', 'RightFinger4Medial', 'RightFinger4Distal', 'RightFinger3Metacarpal', 
                  'RightFinger3Proximal', 'RightFinger3Medial', 'RightFinger3Distal', 'RightFinger2Metacarpal', 
                  'RightFinger2Proximal', 'RightFinger2Medial', 'RightFinger2Distal']


'''
cyrillic_to_latin_mapping = {
    "А": "A",  
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "YO",
    "Ж": "ZH",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "TS",
    "Ч": "CH",
    "Ш": "SH",
    "Щ": "SHCH",
    "Ъ": "HARD",  # No direct equivalent
    "Ы": "YI",
    "Ь": "SOFT",  # No direct equivalent
    "Э": "EE",
    "Ю": "YU",
    "Я": "YA",
    "prob": "prob"
    # The following Cyrillic letters do not have direct Latin equivalents:
    #  Ъ, Ь
}
char_index_map = {
    'A': 0, 'B': 1, 'CH': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'HARD': 8, 'I': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'R': 16, 'S': 17, 'SH': 18,
    'SHCH': 19, 'SOFT': 20, 'T': 21, 'TS': 22, 'U': 23, 'V': 24, 'Y': 25, 'YA': 26,
    'YI': 27, 'YO': 28, 'YU': 29, 'Z': 30, 'ZH': 31, 'EE':32, 'prob': 33
}



cyrillic_to_latin_mapping_old = {
    "А": "A",  
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "YO",
    "Ж": "ZH",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "TS",
    "Ч": "CH",
    "Ш": "SH",
    "Щ": "SHCH",
    "Ъ": "HARD",  # No direct equivalent
    "Ы": "YI",
    "Ь": "SOFT",  # No direct equivalent
    "Э": "EE",
    "Ю": "YU",
    "Я": "YA",
    # The following Cyrillic letters do not have direct Latin equivalents:
    #  Ъ, Ь
}

letter_to_index = {
    "A": 0, "B": 1, "V": 2, "G": 3, "D": 4, "YO": 5, "ZH": 6, 
    "Z": 7, "I": 8, "Y": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, 
    "P": 15, "R": 16, "S": 17, "T": 18, "U": 19, "F": 20, "H": 21, 
    "TS": 22, "CH": 23, "SH": 24, "SHCH": 25, "HARD": 26, 
    "E": 27, "YU": 28, "YA": 29, "SOFT": 30
}
'''

cyrillic_to_latin_mapping = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 
    'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 
    'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 
    'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'PROB': 'PROB'
}

char_index_map = {char: index for index, char in enumerate(cyrillic_to_latin_mapping.keys())}




num_letters = len(cyrillic_to_latin_mapping.keys())  # Th


latin_to_cyrillic_mapping = {value: key for key, value in cyrillic_to_latin_mapping.items()}

ordered_sequence = {
    "673": "A",
    "925": "B",
    "1206": "V",
    "1327": "G",
    "1521": "D",
    "1749": "E",
    "1831": "YO",
    "2074": "ZH",
    "2238": "Z",
    "2364": "I",
    "2548": "Y",
    "2728": "K",
    "2956": "L",
    "3212": "M",
    "3324": "N",
    "3618": "O",
    "3813": "P",
    "3978": "R",
    "4119": "S",
    "4351": "T",
    "4516": "U",
    "4652": "F",
    "4763": "H",
    "5214": "TS",
    "5381": "CH",
    "5553": "SH",
    "5815": "SHCH",
    "6890": "HARD",
#    "6910": "SOFT",
    "7273": "Y",
    "7583": "E",
    "7894": "YU",
    "8024": "YA"
}


joint_to_param_map_1 = {
    # Thumb
    'RightFinger1Metacarpal-RightFinger1Proximal-RightFinger1Distal': "Thumb 02 R Ctrl",
    # Assuming there's a Thumb 02 and Thumb 03 if applicable
    'RightFinger1Proximal-RightFinger1Distal-RightFinger1Tip': "Thumb 03 R Ctrl",  # If you have a tip joint
    
    # Index finger
    'RightFinger2Metacarpal-RightFinger2Proximal-RightFinger2Medial': "Index 01 R Ctrl",
    'RightFinger2Proximal-RightFinger2Medial-RightFinger2Distal': "Index 02 R Ctrl",
    # Assuming there's an Index 03 if applicable, you would map it here
    
    # Middle finger
    'RightFinger3Metacarpal-RightFinger3Proximal-RightFinger3Medial': "Middle 01 R Ctrl",
    'RightFinger3Proximal-RightFinger3Medial-RightFinger3Distal': "Middle 02 R Ctrl",
    # Assuming there's a Middle 03 R Ctrl, it would be added here
    
    # Ring finger
    'RightFinger4Metacarpal-RightFinger4Proximal-RightFinger4Medial': "Ring 01 R Ctrl",
    'RightFinger4Proximal-RightFinger4Medial-RightFinger4Distal': "Ring 02 R Ctrl",
    # Assuming there's a Ring 03 R Ctrl, it would be added here
    
    # Pinky finger
    'RightFinger5Metacarpal-RightFinger5Proximal-RightFinger5Medial': "Pinky 01 R Ctrl",
    'RightFinger5Proximal-RightFinger5Medial-RightFinger5Distal': "Pinky 02 R Ctrl",
    # Assuming there's a Pinky 03 R Ctrl, it would be added here
}

sequence_by_frame = {k:v for v,k in ordered_sequence.items()}

DATA_MIN = -90.38202667236328
DATA_MAX = 161.07981872558594



'''
data = np.load(path)['data']


coordinates_input_gt = {}

for letter in sequence_by_frame:
    coordinates_input_gt[letter] = (data[:,:, int(sequence_by_frame[letter])]  - DATA_MIN)/ (DATA_MAX - DATA_MIN)

print(coordinates_input_gt)
'''

coordinates_input_gt = {'A': np.array([[0.40894533, 0.90193985, 0.42364918],
       [0.41931312, 0.90815461, 0.4336816 ],
       [0.42451419, 0.91573528, 0.44127392],
       [0.39736882, 0.90341426, 0.43604742],
       [0.3940288 , 0.91929849, 0.44613208],
       [0.40622869, 0.91666181, 0.45267299],
       [0.40824974, 0.90923812, 0.44828443],
       [0.40005102, 0.90444534, 0.43346402],
       [0.39912061, 0.92371596, 0.44003549],
       [0.41292774, 0.9196931 , 0.44985685],
       [0.4141149 , 0.90936288, 0.44561202],
       [0.40306116, 0.90457562, 0.43034848],
       [0.40515882, 0.92563522, 0.43259405],
       [0.41724909, 0.92414388, 0.44670882],
       [0.41652185, 0.91230648, 0.4434564 ],
       [0.40592185, 0.90413065, 0.42726786],
       [0.41188878, 0.92507089, 0.42544815],
       [0.42421947, 0.92687437, 0.43865952],
       [0.42362608, 0.91656296, 0.44292166]]), 'B': np.array([[0.41176117, 0.90585246, 0.41585604],
       [0.42261104, 0.90752347, 0.42709225],
       [0.42424854, 0.91341098, 0.43733026],
       [0.40035994, 0.90953011, 0.42795532],
       [0.4025208 , 0.92296328, 0.4413735 ],
       [0.41279178, 0.9223867 , 0.45100406],
       [0.41743001, 0.91525075, 0.4485504 ],
       [0.40352029, 0.9098346 , 0.42575289],
       [0.40965635, 0.92595779, 0.43660637],
       [0.4218716 , 0.92856116, 0.44874265],
       [0.42490345, 0.91775272, 0.4490998 ],
       [0.406679  , 0.90931118, 0.42283174],
       [0.41680495, 0.92661436, 0.42997422],
       [0.4258598 , 0.94019006, 0.43899289],
       [0.43393042, 0.93765399, 0.44791833],
       [0.40946587, 0.90833434, 0.41980667],
       [0.42354363, 0.92472295, 0.4230686 ],
       [0.43355104, 0.927487  , 0.43796996],
       [0.43929767, 0.9270677 , 0.44754309]]), 'V':np.array([[0.40059472, 0.92034025, 0.40831358],
       [0.41456318, 0.92673189, 0.4115989 ],
       [0.42150885, 0.93475954, 0.41702803],
       [0.39494186, 0.91886293, 0.42430635],
       [0.40000956, 0.93018363, 0.43884343],
       [0.40378423, 0.94085615, 0.4472353 ],
       [0.40757364, 0.94737492, 0.45188309],
       [0.39674707, 0.92035536, 0.42123305],
       [0.40373598, 0.9357124 , 0.43266645],
       [0.41034015, 0.94954333, 0.44093492],
       [0.41718755, 0.95610348, 0.44695319],
       [0.39840239, 0.92116987, 0.41731119],
       [0.40712467, 0.93930692, 0.42423349],
       [0.41691837, 0.95236539, 0.43324384],
       [0.42609282, 0.95794926, 0.43923383],
       [0.3997702 , 0.92146727, 0.41332215],
       [0.41051666, 0.94041094, 0.41507105],
       [0.42304643, 0.95085123, 0.4230612 ],
       [0.43183089, 0.95566009, 0.42801622]]), 'G':np.array([[0.31874476, 0.94475505, 0.45476391],
       [0.32510995, 0.95727704, 0.44773198],
       [0.33263977, 0.96630433, 0.44972469],
       [0.32877203, 0.93107721, 0.45627237],
       [0.34780809, 0.93152983, 0.45788182],
       [0.34793478, 0.93622212, 0.44459499],
       [0.34007244, 0.93664075, 0.44053744],
       [0.32771097, 0.93479206, 0.4562001 ],
       [0.34704109, 0.94047987, 0.45926645],
       [0.34760369, 0.94376796, 0.44217403],
       [0.33750221, 0.94081185, 0.43825442],
       [0.32557605, 0.93855285, 0.45591159],
       [0.34336876, 0.94951153, 0.45994593],
       [0.34730093, 0.950634  , 0.44175518],
       [0.33657542, 0.94739749, 0.43668379],
       [0.3230241 , 0.94190064, 0.45552247],
       [0.33780869, 0.95757947, 0.45912805],
       [0.35230052, 0.9623453 , 0.44927349],
       [0.36085616, 0.96441165, 0.44239051]]), 'D': np.array([[0.40827192, 0.91340255, 0.41945735],
       [0.41670631, 0.91337372, 0.43270969],
       [0.41453826, 0.91662771, 0.44397344],
       [0.39340343, 0.91197249, 0.42762978],
       [0.38952802, 0.9194491 , 0.44478344],
       [0.39908028, 0.91207753, 0.45206263],
       [0.40472816, 0.90576332, 0.44947674],
       [0.39699219, 0.9131642 , 0.42683488],
       [0.39723295, 0.9242008 , 0.44396802],
       [0.40742117, 0.9149019 , 0.4545986 ],
       [0.41071915, 0.90518158, 0.45003997],
       [0.40091213, 0.91385795, 0.42512105],
       [0.40584137, 0.927485  , 0.4407081 ],
       [0.41135039, 0.93344508, 0.45749361],
       [0.41610572, 0.93591604, 0.46856207],
       [0.4045778 , 0.91416081, 0.42303708],
       [0.41451575, 0.92849988, 0.43619146],
       [0.4181427 , 0.9378728 , 0.45131863],
       [0.42060827, 0.94287741, 0.4609995 ]]), 'E':np.array([[0.37222266, 0.90285728, 0.47313031],
       [0.37525678, 0.9074634 , 0.48783886],
       [0.37592352, 0.91318021, 0.49828082],
       [0.35553382, 0.90604099, 0.47424929],
       [0.34784974, 0.92206643, 0.48127118],
       [0.35168295, 0.92229549, 0.49482946],
       [0.35660193, 0.91598541, 0.49862972],
       [0.35934739, 0.90665028, 0.4743784 ],
       [0.35618917, 0.92566962, 0.48098878],
       [0.35963451, 0.92369138, 0.49794447],
       [0.36387943, 0.91341832, 0.49633617],
       [0.36366695, 0.90632649, 0.47423282],
       [0.36586642, 0.92667625, 0.48006353],
       [0.36648073, 0.92808003, 0.49864517],
       [0.3677611 , 0.9158533 , 0.49833159],
       [0.36779398, 0.90546064, 0.47393358],
       [0.37555702, 0.92511276, 0.47949593],
       [0.37850001, 0.94113765, 0.48751966],
       [0.380136  , 0.94895388, 0.49533467]]), 'YO':np.array([[0.40269353, 0.91764004, 0.40211652],
       [0.41324374, 0.92408975, 0.41180458],
       [0.41862277, 0.93094693, 0.41994129],
       [0.39297611, 0.92047223, 0.41580798],
       [0.39176653, 0.93708123, 0.42518045],
       [0.40400058, 0.93850442, 0.43202703],
       [0.41186369, 0.93443023, 0.43186022],
       [0.39533017, 0.92120131, 0.41283172],
       [0.39618856, 0.94084723, 0.41818961],
       [0.40928498, 0.94414078, 0.42918577],
       [0.41782949, 0.93708165, 0.43100264],
       [0.39789532, 0.92098328, 0.40934503],
       [0.40125018, 0.94199227, 0.40990785],
       [0.41363788, 0.94766424, 0.42263573],
       [0.42161038, 0.93952986, 0.42727271],
       [0.40029059, 0.92020202, 0.40595032],
       [0.40692127, 0.94064128, 0.40199025],
       [0.41935782, 0.9464152 , 0.41389962],
       [0.42486551, 0.94224039, 0.42267916]]), 'ZH':np.array([[0.41346167, 0.89935129, 0.42276647],
       [0.42059518, 0.9029236 , 0.43629849],
       [0.42107903, 0.90823398, 0.44696276],
       [0.400497  , 0.90062721, 0.43372984],
       [0.39815871, 0.91475008, 0.44638861],
       [0.40897127, 0.90987793, 0.45399935],
       [0.41608307, 0.90621017, 0.4577974 ],
       [0.40366943, 0.90160538, 0.43175247],
       [0.4047447 , 0.91920122, 0.44198179],
       [0.41606972, 0.91365   , 0.45399044],
       [0.42331937, 0.90782275, 0.46028544],
       [0.40710362, 0.90175041, 0.42911242],
       [0.41207262, 0.92131089, 0.4358687 ],
       [0.42202172, 0.91656509, 0.45090595],
       [0.42867279, 0.90909303, 0.45805865],
       [0.41029741, 0.90136017, 0.42637029],
       [0.41964553, 0.92084074, 0.42961531],
       [0.42779523, 0.91695089, 0.44537254],
       [0.43234845, 0.9124046 , 0.45450726]]), 'Z':np.array([[0.41540728, 0.89514408, 0.42378273],
       [0.42373833, 0.89932356, 0.43642749],
       [0.42442647, 0.90579147, 0.44642014],
       [0.4019254 , 0.89614555, 0.43413366],
       [0.39907605, 0.90981749, 0.44717691],
       [0.40957141, 0.90680799, 0.45608551],
       [0.41595885, 0.90068316, 0.45646361],
       [0.40519982, 0.89716419, 0.43235267],
       [0.40590828, 0.91436184, 0.44326811],
       [0.41715305, 0.91172207, 0.45630135],
       [0.42224998, 0.90182789, 0.4578078 ],
       [0.40876181, 0.89737183, 0.42989226],
       [0.41354724, 0.91662146, 0.43760601],
       [0.42342966, 0.91508224, 0.45334114],
       [0.42772305, 0.90399212, 0.45647265],
       [0.41208445, 0.89705084, 0.42729827],
       [0.42141701, 0.91630598, 0.43172029],
       [0.430176  , 0.9277867 , 0.44273441],
       [0.43584593, 0.93405085, 0.45004578]]), 'I':np.array([[0.41301143, 0.89556411, 0.42567932],
       [0.42065032, 0.9037278 , 0.43671404],
       [0.42173267, 0.91263737, 0.44456318],
       [0.39908551, 0.89997623, 0.43442621],
       [0.39544019, 0.91747317, 0.44118907],
       [0.39428699, 0.93135828, 0.44329792],
       [0.39443994, 0.94008339, 0.444816  ],
       [0.40241406, 0.90034954, 0.43249938],
       [0.40235043, 0.92041562, 0.43607056],
       [0.41261835, 0.93026507, 0.44611234],
       [0.41946017, 0.93521585, 0.45351643],
       [0.40607328, 0.89972453, 0.43026239],
       [0.41018192, 0.92060676, 0.43022677],
       [0.42088026, 0.92227456, 0.44540536],
       [0.42727388, 0.91695119, 0.45446147],
       [0.4095085 , 0.89855231, 0.42809519],
       [0.41829944, 0.9183395 , 0.42516319],
       [0.42728977, 0.92420286, 0.43981358],
       [0.43255476, 0.92366371, 0.44965397]]), 'Y':np.array([[0.36614237, 0.89840024, 0.47858238],
       [0.36707906, 0.90735976, 0.49145146],
       [0.36282923, 0.91743815, 0.49619757],
       [0.34945323, 0.90176266, 0.47831281],
       [0.34146355, 0.91820273, 0.4838855 ],
       [0.33857022, 0.93199294, 0.48406357],
       [0.33705628, 0.94059765, 0.48552039],
       [0.35325417, 0.90235005, 0.47868574],
       [0.34984169, 0.92172388, 0.48401641],
       [0.35071506, 0.92412822, 0.50124243],
       [0.35216018, 0.91404728, 0.50597842],
       [0.35756904, 0.90198712, 0.47887206],
       [0.35957121, 0.92260763, 0.48374402],
       [0.35869165, 0.92674221, 0.50190324],
       [0.35774545, 0.91484425, 0.50486512],
       [0.36169689, 0.901076  , 0.47892049],
       [0.36925926, 0.92094445, 0.483967  ],
       [0.37263516, 0.93680847, 0.49213851],
       [0.37479092, 0.9459496 , 0.49819132]]), 'K':np.array([[0.35258483, 0.9522214 , 0.41110727],
       [0.3682752 , 0.95270702, 0.41169093],
       [0.37713806, 0.94827705, 0.41832313],
       [0.35540189, 0.93666897, 0.41743915],
       [0.36863799, 0.93236618, 0.43053336],
       [0.37972919, 0.93467677, 0.42215365],
       [0.38344795, 0.93750666, 0.41462918],
       [0.35566086, 0.94050081, 0.41701352],
       [0.36962875, 0.94130676, 0.43183431],
       [0.38385363, 0.94126647, 0.42178791],
       [0.38811868, 0.94301892, 0.41154682],
       [0.35526957, 0.94463029, 0.41575683],
       [0.36892266, 0.95099857, 0.43078972],
       [0.38436756, 0.94945534, 0.44111925],
       [0.39573435, 0.94883859, 0.44577176],
       [0.35451994, 0.94845994, 0.41413093],
       [0.36747947, 0.96008848, 0.42733079],
       [0.37901376, 0.96433653, 0.44070064],
       [0.38663179, 0.96671902, 0.44851943]]), 'L':np.array([[0.38742179, 0.93183626, 0.40712059],
       [0.39558571, 0.92275624, 0.41700331],
       [0.39523785, 0.9198897 , 0.42857149],
       [0.37388237, 0.92715223, 0.41632108],
       [0.37370154, 0.92502781, 0.4353111 ],
       [0.38133711, 0.91356299, 0.4323397 ],
       [0.38434225, 0.91085889, 0.42445857],
       [0.37744565, 0.92841074, 0.41551458],
       [0.38179585, 0.92916068, 0.43541234],
       [0.38905339, 0.91378829, 0.43163134],
       [0.3911558 , 0.91177492, 0.42078385],
       [0.38112572, 0.92969352, 0.41361822],
       [0.39020255, 0.93323434, 0.43253972],
       [0.39510033, 0.91808149, 0.44223656],
       [0.39862924, 0.90669755, 0.44526703],
       [0.38444636, 0.93084068, 0.41126685],
       [0.39816452, 0.9359942 , 0.42747372],
       [0.40501427, 0.92148262, 0.43597858],
       [0.40863173, 0.91162558, 0.43979927]]), 'M':np.array([[0.38184866, 0.93375528, 0.40186244],
       [0.3910825 , 0.92565676, 0.41165601],
       [0.39255063, 0.92324532, 0.42324008],
       [0.36890094, 0.93167419, 0.41272206],
       [0.37044899, 0.93151575, 0.43176792],
       [0.37802726, 0.91963672, 0.43158879],
       [0.38234776, 0.9134432 , 0.42695966],
       [0.37247783, 0.93241042, 0.41145911],
       [0.37891236, 0.93464194, 0.43066912],
       [0.38648768, 0.92134312, 0.43897788],
       [0.39121857, 0.91132388, 0.44081521],
       [0.37606044, 0.93304665, 0.40910438],
       [0.38737145, 0.93737195, 0.42660588],
       [0.39552564, 0.92846772, 0.44081329],
       [0.40087095, 0.91944152, 0.4472309 ],
       [0.37922601, 0.93354587, 0.40634727],
       [0.39502157, 0.93863586, 0.42056   ],
       [0.40415051, 0.92759483, 0.43172226],
       [0.40936162, 0.91917847, 0.43690424]]), 'N':np.array([[0.41504568, 0.89838204, 0.41497199],
       [0.4227693 , 0.90742286, 0.42523715],
       [0.42305382, 0.91620828, 0.433293  ],
       [0.40159777, 0.90402665, 0.4237582 ],
       [0.39903015, 0.92205817, 0.42954063],
       [0.39514062, 0.93559723, 0.42916778],
       [0.39408988, 0.94439187, 0.42909096],
       [0.40488934, 0.90411051, 0.42173588],
       [0.40594505, 0.92432737, 0.42409603],
       [0.4167971 , 0.93258026, 0.43493121],
       [0.42353239, 0.92577327, 0.44079995],
       [0.40845187, 0.90316359, 0.41945634],
       [0.41362003, 0.92376268, 0.41807155],
       [0.41858555, 0.94161933, 0.42009869],
       [0.42406883, 0.95224367, 0.42297724],
       [0.41176464, 0.901687  , 0.41728449],
       [0.4214754 , 0.92077806, 0.41296667],
       [0.4304859 , 0.9362507 , 0.41600761],
       [0.43713452, 0.94480025, 0.41875433]]), 'O':np.array([[0.40619829, 0.90037569, 0.43199382],
       [0.41485837, 0.90938569, 0.44151155],
       [0.41889399, 0.918133  , 0.44853716],
       [0.39290599, 0.90448994, 0.44180676],
       [0.38889217, 0.92209677, 0.44805623],
       [0.38314061, 0.93488035, 0.44949518],
       [0.38161092, 0.94348319, 0.45094675],
       [0.39601548, 0.90493898, 0.43955717],
       [0.39511896, 0.92513218, 0.44217216],
       [0.39776301, 0.94182067, 0.446389  ],
       [0.40179726, 0.95119808, 0.45107213],
       [0.3994829 , 0.90439401, 0.43701462],
       [0.40235404, 0.92543515, 0.43560858],
       [0.40998362, 0.94028794, 0.44390373],
       [0.41763241, 0.94640033, 0.4513445 ],
       [0.40276581, 0.90329496, 0.43458844],
       [0.4100878 , 0.92334247, 0.42990909],
       [0.41900874, 0.93133668, 0.44356004],
       [0.42417007, 0.92878356, 0.45313535]]), 'P':np.array([[0.40179054, 0.92038685, 0.41134255],
       [0.40917931, 0.91456282, 0.42392235],
       [0.40871215, 0.91499814, 0.43582837],
       [0.38685074, 0.91992659, 0.41949709],
       [0.38428098, 0.92408684, 0.43797019],
       [0.39187264, 0.91261   , 0.44100718],
       [0.39532392, 0.90678026, 0.43530122],
       [0.39060284, 0.92059511, 0.41885982],
       [0.39273131, 0.92743348, 0.4379416 ],
       [0.40015562, 0.91187431, 0.44040596],
       [0.40189433, 0.90626514, 0.43083225],
       [0.39459627, 0.92092497, 0.41720808],
       [0.40181958, 0.9297383 , 0.43518303],
       [0.40761915, 0.91674822, 0.44723496],
       [0.41125478, 0.9058277 , 0.45156568],
       [0.39827092, 0.92100567, 0.41511948],
       [0.41056671, 0.93012568, 0.4307095 ],
       [0.41431235, 0.91777991, 0.44349193],
       [0.41649296, 0.90887136, 0.44987374]]), 'R':np.array([[0.41189047, 0.90178961, 0.42284838],
       [0.41851991, 0.90984261, 0.43459421],
       [0.41947748, 0.91751152, 0.44367342],
       [0.39721583, 0.90445189, 0.43106257],
       [0.39167462, 0.92114505, 0.43853248],
       [0.38836545, 0.93369575, 0.4440191 ],
       [0.38774505, 0.94169791, 0.44776551],
       [0.40058576, 0.90522618, 0.42933766],
       [0.39852786, 0.92497192, 0.43395024],
       [0.40654131, 0.93555797, 0.44521967],
       [0.41287542, 0.94001488, 0.45335334],
       [0.40439022, 0.90505451, 0.42726856],
       [0.40658231, 0.92618516, 0.42854907],
       [0.41645273, 0.93115161, 0.44356689],
       [0.4225806 , 0.92343853, 0.45092811],
       [0.40801822, 0.90431488, 0.42522838],
       [0.41510692, 0.92493418, 0.42381191],
       [0.42285583, 0.93829314, 0.43336857],
       [0.42847324, 0.94530135, 0.44001492]]), 'S':np.array([[0.4124902 , 0.90383453, 0.41392806],
       [0.42144292, 0.91099141, 0.42467014],
       [0.42538037, 0.91834921, 0.43318604],
       [0.39946767, 0.90854873, 0.4238324 ],
       [0.39713233, 0.92595239, 0.43137058],
       [0.40605167, 0.93375892, 0.43899134],
       [0.41264203, 0.93764022, 0.44345878],
       [0.40265721, 0.90884224, 0.42167095],
       [0.40376909, 0.92876317, 0.42583331],
       [0.41174522, 0.94161872, 0.43445861],
       [0.42006047, 0.94344775, 0.44178342],
       [0.40610784, 0.90813507, 0.41914554],
       [0.41114493, 0.92881117, 0.41942133],
       [0.41821519, 0.94336146, 0.428691  ],
       [0.42640889, 0.94389181, 0.43784591],
       [0.40931571, 0.90688876, 0.41669035],
       [0.41875227, 0.92636623, 0.41369201],
       [0.42439252, 0.94201405, 0.4209839 ],
       [0.42999996, 0.94691168, 0.42931541]]), 'T':np.array([[0.40366801, 0.91767074, 0.42114542],
       [0.41212554, 0.91050476, 0.43227567],
       [0.41180401, 0.90945839, 0.44414845],
       [0.38990884, 0.91466488, 0.4307139 ],
       [0.38916303, 0.91636933, 0.44973245],
       [0.39749767, 0.90543837, 0.45283428],
       [0.40147543, 0.90065416, 0.44653001],
       [0.39347928, 0.91580343, 0.42977213],
       [0.39719719, 0.92058965, 0.44923174],
       [0.40419998, 0.90597758, 0.45561351],
       [0.40835653, 0.89563292, 0.45425247],
       [0.39720012, 0.91675046, 0.42776137],
       [0.40563701, 0.92416433, 0.44583908],
       [0.41091245, 0.90841572, 0.45431126],
       [0.41308559, 0.89636648, 0.45316094],
       [0.40057764, 0.91746716, 0.42532202],
       [0.41371741, 0.92600591, 0.44054849],
       [0.41714906, 0.91110173, 0.45034277],
       [0.41844506, 0.900278  , 0.45279457]]), 'U':np.array([[0.40524584, 0.91888587, 0.41138759],
       [0.41597647, 0.92213123, 0.42239148],
       [0.42063468, 0.93041088, 0.42959624],
       [0.39086495, 0.91741734, 0.42038421],
       [0.38614613, 0.92824374, 0.4354071 ],
       [0.38473304, 0.93576173, 0.44724163],
       [0.38507485, 0.94045475, 0.45474593],
       [0.39422552, 0.91887233, 0.41915093],
       [0.39304641, 0.93368944, 0.43309618],
       [0.40241982, 0.92661879, 0.44595793],
       [0.40568349, 0.91604997, 0.44401089],
       [0.39797556, 0.91967137, 0.41713021],
       [0.40102624, 0.93725641, 0.42872375],
       [0.40855411, 0.9283009 , 0.44324121],
       [0.41009007, 0.91638074, 0.44063729],
       [0.40152753, 0.91996865, 0.41485712],
       [0.4093926 , 0.93834781, 0.42367522],
       [0.41461484, 0.93141496, 0.43962835],
       [0.41505658, 0.92033291, 0.44098341]]), 'F':np.array([[0.36744636, 0.94399478, 0.40460434],
       [0.38114587, 0.95151307, 0.40300353],
       [0.38772692, 0.96043047, 0.40740019],
       [0.37068308, 0.93834702, 0.42033752],
       [0.38031326, 0.94801036, 0.4337183 ],
       [0.39365783, 0.95013114, 0.42971853],
       [0.40166808, 0.95102005, 0.42604415],
       [0.37048089, 0.94077423, 0.41733763],
       [0.37958095, 0.9553096 , 0.42835218],
       [0.39592891, 0.95385339, 0.42252959],
       [0.40571005, 0.95102879, 0.41778698],
       [0.3698443 , 0.94266139, 0.41348822],
       [0.37776721, 0.96106471, 0.42066408],
       [0.39597052, 0.95827541, 0.41775123],
       [0.40613996, 0.95308433, 0.41318339],
       [0.36901237, 0.9439912 , 0.40956251],
       [0.37600245, 0.96458653, 0.41165536],
       [0.39394549, 0.96178352, 0.41182459],
       [0.40414338, 0.95754458, 0.41012793]]), 'H':np.array([[0.39898777, 0.92351711, 0.39641037],
       [0.41151514, 0.92990863, 0.40340878],
       [0.41631172, 0.93830394, 0.41038535],
       [0.38973273, 0.92333944, 0.41070087],
       [0.38949104, 0.93724033, 0.42381092],
       [0.40186173, 0.93893792, 0.43034218],
       [0.41026693, 0.93620858, 0.43094097],
       [0.39204632, 0.92467313, 0.40790806],
       [0.39391158, 0.94240672, 0.41777967],
       [0.40924848, 0.94130076, 0.42595497],
       [0.41596344, 0.93230921, 0.42550416],
       [0.39451618, 0.92520305, 0.4043862 ],
       [0.39882284, 0.94531725, 0.40984701],
       [0.4136565 , 0.9445383 , 0.42111561],
       [0.4199212 , 0.93409733, 0.42283875],
       [0.39679328, 0.92517101, 0.40082452],
       [0.40423934, 0.9456962 , 0.4016499 ],
       [0.41562046, 0.95719342, 0.40990344],
       [0.42369801, 0.96299634, 0.41499494]]), 'TS':np.array([[0.41074902, 0.90209258, 0.42252701],
       [0.41807507, 0.91041483, 0.43365508],
       [0.41818253, 0.91855303, 0.44236822],
       [0.397257  , 0.90425553, 0.43268539],
       [0.39246711, 0.92065154, 0.44125241],
       [0.40392742, 0.92022818, 0.44944113],
       [0.41090499, 0.91492399, 0.45071948],
       [0.40035233, 0.90514007, 0.43054814],
       [0.39857856, 0.92472641, 0.43589939],
       [0.41192192, 0.92141193, 0.44658808],
       [0.41528645, 0.91089402, 0.4445398 ],
       [0.40384906, 0.9050962 , 0.42798777],
       [0.40581735, 0.92623304, 0.42950809],
       [0.41790576, 0.92722753, 0.44366814],
       [0.42586253, 0.92224913, 0.45161393],
       [0.40718487, 0.90448029, 0.42546493],
       [0.41365433, 0.92527114, 0.42364881],
       [0.42399897, 0.92784228, 0.4383531 ],
       [0.42961539, 0.92560712, 0.44775011]]), 'CH':np.array([[0.41032942, 0.90334011, 0.42263867],
       [0.41779426, 0.91119827, 0.43400923],
       [0.41796807, 0.91909424, 0.44294143],
       [0.39682807, 0.90520736, 0.43284323],
       [0.39212302, 0.92130088, 0.44200988],
       [0.40358827, 0.92051616, 0.45016495],
       [0.41055988, 0.91517314, 0.4513069 ],
       [0.39993389, 0.90615149, 0.4307471 ],
       [0.39827263, 0.92553425, 0.43682627],
       [0.41158892, 0.92170653, 0.44737634],
       [0.41516984, 0.91134609, 0.44493091],
       [0.40343616, 0.90618116, 0.42819411],
       [0.40553558, 0.92723287, 0.43051098],
       [0.41746325, 0.92793452, 0.44482405],
       [0.42543176, 0.92272936, 0.45261118],
       [0.40677379, 0.90563892, 0.4256568 ],
       [0.41337975, 0.92644105, 0.4246359 ],
       [0.42358734, 0.92869417, 0.43948741],
       [0.42910952, 0.9262382 , 0.44888516]]), 'SH':np.array([[0.33360117, 0.93714458, 0.45556152],
       [0.33771421, 0.95133637, 0.45022868],
       [0.3465446 , 0.95925109, 0.45147045],
       [0.35044203, 0.93476931, 0.45636721],
       [0.36318217, 0.94805873, 0.46149036],
       [0.36223633, 0.95902694, 0.45269396],
       [0.35761271, 0.95954934, 0.44515708],
       [0.34701077, 0.93649797, 0.45677878],
       [0.35615369, 0.95308142, 0.46431556],
       [0.35743551, 0.96840678, 0.4561443 ],
       [0.3578875 , 0.97718881, 0.44915759],
       [0.34280795, 0.9375566 , 0.45679314],
       [0.34706792, 0.95629414, 0.4659423 ],
       [0.35031508, 0.97194251, 0.4563397 ],
       [0.35142185, 0.9804775 , 0.4475557 ],
       [0.33861882, 0.93807438, 0.45655882],
       [0.33737801, 0.95792632, 0.46560144],
       [0.3421242 , 0.97377651, 0.45811249],
       [0.34483595, 0.98285016, 0.45218296]]), 'SHCH':np.array([[0.33280909, 0.91955086, 0.47516157],
       [0.33769654, 0.9342835 , 0.47274752],
       [0.34672978, 0.94137825, 0.47594528],
       [0.34945226, 0.91611654, 0.47621791],
       [0.36251712, 0.92744683, 0.48434814],
       [0.36226846, 0.93972355, 0.47743483],
       [0.35793105, 0.94160046, 0.46994355],
       [0.34608962, 0.91792681, 0.47680687],
       [0.35559418, 0.93223038, 0.48778314],
       [0.35780104, 0.94833785, 0.48154148],
       [0.3585678 , 0.95761266, 0.47525401],
       [0.34194675, 0.91919952, 0.47684418],
       [0.34659709, 0.93558418, 0.48960616],
       [0.35071174, 0.95223402, 0.48229327],
       [0.35238071, 0.96168346, 0.47460217],
       [0.33780514, 0.91998861, 0.4765341 ],
       [0.33702698, 0.93779713, 0.48916982],
       [0.34278957, 0.95417075, 0.48382787],
       [0.34620325, 0.9637128 , 0.47912254]]), 'HARD':np.array([[0.33640894, 0.94272711, 0.44974298],
       [0.34689058, 0.95084329, 0.45817077],
       [0.35000719, 0.95847511, 0.46678487],
       [0.32958596, 0.93429758, 0.46286915],
       [0.32635154, 0.94169081, 0.48019098],
       [0.33921465, 0.94479491, 0.48503637],
       [0.34662703, 0.94255921, 0.48073362],
       [0.33095852, 0.93725077, 0.46078925],
       [0.32801585, 0.94984631, 0.47654047],
       [0.34248054, 0.95307141, 0.48568653],
       [0.35239637, 0.94806055, 0.48404059],
       [0.33265472, 0.93981688, 0.45773599],
       [0.33065502, 0.95688359, 0.47029304],
       [0.34437581, 0.96073479, 0.48231526],
       [0.35517392, 0.95495304, 0.48341157],
       [0.33435304, 0.94184111, 0.45443596],
       [0.33457716, 0.9619973 , 0.4628673 ],
       [0.34005408, 0.97514455, 0.47413635],
       [0.3443121 , 0.98282777, 0.48104154]]), 
       'YU':np.array([[0.36436671, 0.89820115, 0.47924104],
       [0.36383751, 0.90565719, 0.49305741],
       [0.35915175, 0.91420511, 0.49992307],
       [0.34785443, 0.90229137, 0.47851994],
       [0.34054267, 0.91933375, 0.48313097],
       [0.34093934, 0.93182212, 0.48964678],
       [0.34149779, 0.93968994, 0.4936769 ],
       [0.35167231, 0.9027256 , 0.47892796],
       [0.34907009, 0.92247735, 0.48322949],
       [0.35123512, 0.92264307, 0.5005085 ],
       [0.35260987, 0.91917027, 0.51110056],
       [0.35596287, 0.90217923, 0.47920567],
       [0.35883278, 0.92290861, 0.48307948],
       [0.3581852 , 0.92297924, 0.50171281],
       [0.35900499, 0.91866972, 0.51320133],
       [0.36004361, 0.90108729, 0.47936979],
       [0.36843037, 0.92082551, 0.48355015],
       [0.36630527, 0.92387385, 0.5013274 ],
       [0.36440505, 0.9218123 , 0.51214327]]), 'YA':np.array([[0.37481725, 0.90560664, 0.473956  ],
       [0.37074309, 0.90490135, 0.48911081],
       [0.36229108, 0.90940766, 0.4962113 ],
       [0.35837587, 0.90894351, 0.47104868],
       [0.34710317, 0.91937938, 0.48241468],
       [0.34644467, 0.91264799, 0.49477703],
       [0.35032384, 0.90515433, 0.4974701 ],
       [0.36185983, 0.90911014, 0.47271166],
       [0.35464128, 0.92196412, 0.4867856 ],
       [0.35301103, 0.91208639, 0.50103524],
       [0.35743654, 0.90189167, 0.49941539],
       [0.36601314, 0.90859418, 0.4738379 ],
       [0.36393762, 0.9225904 , 0.48973597],
       [0.36059403, 0.92617679, 0.50772441],
       [0.35878611, 0.92475669, 0.51980518],
       [0.370095  , 0.90771437, 0.47449823],
       [0.37342627, 0.92085094, 0.49163716],
       [0.36741829, 0.92064687, 0.50877488],
       [0.36415425, 0.91916469, 0.51935756]])}
       
'''
'SOFT':np.array([[0.37336398, 0.91889048, 0.46123185],
[0.38096181, 0.92367809, 0.47412044],
[0.38321978, 0.93066088, 0.48351741],
[0.35759563, 0.9167234 , 0.46727906],
[0.34890894, 0.92832906, 0.47972966],
[0.35759522, 0.92626459, 0.49063193],
[0.3637704 , 0.92021203, 0.48871126],
[0.36100182, 0.91841116, 0.46658576],
[0.35564412, 0.93424812, 0.47824301],
[0.36368282, 0.93275745, 0.49361951],
[0.36893324, 0.92292481, 0.49224345],
[0.3650094 , 0.91941834, 0.46527829],
[0.364026  , 0.93823221, 0.47517873],
[0.3693419 , 0.93853131, 0.49304707],
[0.37315712, 0.92685228, 0.4935714 ],
[0.36892041, 0.91988412, 0.46374246],
[0.37314011, 0.93974485, 0.47181426],
[0.37616107, 0.95441196, 0.48208984],
[0.37827825, 0.96279852, 0.48916284]])
'''


transitions = ['АБ',
 'АВ',
 'АГ',
 'АД',
 'АЕ',
 'АЁ',
 'АЖ',
 'АЗ',
 'АИ',
 'АЙ',
 'АК',
 'АЛ',
 'АМ',
 'АН',
 'АО',
 'АП',
 'АР',
 'АС',
 'АТ',
 'АУ',
 'АФ',
 'АХ',
 'АЦ',
 'АЧ',
 'АШ',
 'АЩ',
 'АЪ',
 'АЫ',
 'АЬ',
 'АЭ',
 'АЮ',
 'АЯ',
 'БВ',
 'БГ',
 'БД',
 'БЕ',
 'БЁ',
 'БЖ',
 'БЗ',
 'БИ',
 'БЙ',
 'БК',
 'БЛ',
 'БМ',
 'БН',
 'БО',
 'БП',
 'БР',
 'БС',
 'БТ',
 'БУ',
 'БФ',
 'БХ',
 'БЦ',
 'БЧ',
 'БШ',
 'БЩ',
 'БЪ',
 'БЫ',
 'БЬ',
 'БЭ',
 'БЮ',
 'БЯ',
 'ВГ',
 'ВД',
 'ВЕ',
 'ВЁ',
 'ВЖ',
 'ВЗ',
 'ВИ',
 'ВЙ',
 'ВК',
 'ВЛ',
 'ВМ',
 'ВН',
 'ВО',
 'ВП',
 'ВР',
 'ВС',
 'ВТ',
 'ВУ',
 'ВФ',
 'ВХ',
 'ВЦ',
 'ВЧ',
 'ВШ',
 'ВЩ',
 'ВЪ',
 'ВЫ',
 'ВЬ',
 'ВЭ',
 'ВЮ',
 'ВЯ',
 'ГД',
 'ГЕ',
 'ГЁ',
 'ГЖ',
 'ГЗ',
 'ГИ',
 'ГЙ',
 'ГК',
 'ГЛ',
 'ГМ',
 'ГН',
 'ГО',
 'ГП',
 'ГР',
 'ГС',
 'ГТ',
 'ГУ',
 'ГФ',
 'ГХ',
 'ГЦ',
 'ГЧ',
 'ГШ',
 'ГЩ',
 'ГЪ',
 'ГЫ',
 'ГЬ',
 'ГЭ',
 'ГЮ',
 'ГЯ',
 'ДЕ',
 'ДЁ',
 'ДЖ',
 'ДЗ',
 'ДИ',
 'ДЙ',
 'ДК',
 'ДЛ',
 'ДМ',
 'ДН',
 'ДО',
 'ДП',
 'ДР',
 'ДС',
 'ДТ',
 'ДУ',
 'ДФ',
 'ДХ',
 'ДЦ',
 'ДЧ',
 'ДШ',
 'ДЩ',
 'ДЪ',
 'ДЫ',
 'ДЬ',
 'ДЭ',
 'ДЮ',
 'ДЯ',
 'ЕЁ',
 'ЕЖ',
 'ЕЗ',
 'ЕИ',
 'ЕЙ',
 'ЕК',
 'ЕЛ',
 'ЕМ',
 'ЕН',
 'ЕО',
 'ЕП',
 'ЕР',
 'ЕС',
 'ЕТ',
 'ЕУ',
 'ЕФ',
 'ЕХ',
 'ЕЦ',
 'ЕЧ',
 'ЕШ',
 'ЕЩ',
 'ЕЪ',
 'ЕЫ',
 'ЕЬ',
 'ЕЭ',
 'ЕЮ',
 'ЕЯ',
 'ЁЖ',
 'ЁЗ',
 'ЁИ',
 'ЁЙ',
 'ЁК',
 'ЁЛ',
 'ЁМ',
 'ЁН',
 'ЁО',
 'ЁП',
 'ЁР',
 'ЁС',
 'ЁТ',
 'ЁУ',
 'ЁФ',
 'ЁХ',
 'ЁЦ',
 'ЁЧ',
 'ЁШ',
 'ЁЩ',
 'ЁЪ',
 'ЁЫ',
 'ЁЬ',
 'ЁЭ',
 'ЁЮ',
 'ЁЯ',
 'ЖЗ',
 'ЖИ',
 'ЖЙ',
 'ЖК',
 'ЖЛ',
 'ЖМ',
 'ЖН',
 'ЖО',
 'ЖП',
 'ЖР',
 'ЖС',
 'ЖТ',
 'ЖУ',
 'ЖФ',
 'ЖХ',
 'ЖЦ',
 'ЖЧ',
 'ЖШ',
 'ЖЩ',
 'ЖЪ',
 'ЖЫ',
 'ЖЬ',
 'ЖЭ',
 'ЖЮ',
 'ЖЯ',
 'ЗИ',
 'ЗЙ',
 'ЗК',
 'ЗЛ',
 'ЗМ',
 'ЗН',
 'ЗО',
 'ЗП',
 'ЗР',
 'ЗС',
 'ЗТ',
 'ЗУ',
 'ЗФ',
 'ЗХ',
 'ЗЦ',
 'ЗЧ',
 'ЗШ',
 'ЗЩ',
 'ЗЪ',
 'ЗЫ',
 'ЗЬ',
 'ЗЭ',
 'ЗЮ',
 'ЗЯ',
 'ИЙ',
 'ИК',
 'ИЛ',
 'ИМ',
 'ИН',
 'ИО',
 'ИП',
 'ИР',
 'ИС',
 'ИТ',
 'ИУ',
 'ИФ',
 'ИХ',
 'ИЦ',
 'ИЧ',
 'ИШ',
 'ИЩ',
 'ИЪ',
 'ИЫ',
 'ИЬ',
 'ИЭ',
 'ИЮ',
 'ИЯ',
 'ЙК',
 'ЙЛ',
 'ЙМ',
 'ЙН',
 'ЙО',
 'ЙП',
 'ЙР',
 'ЙС',
 'ЙТ',
 'ЙУ',
 'ЙФ',
 'ЙХ',
 'ЙЦ',
 'ЙЧ',
 'ЙШ',
 'ЙЩ',
 'ЙЪ',
 'ЙЫ',
 'ЙЬ',
 'ЙЭ',
 'ЙЮ',
 'ЙЯ',
 'КЛ',
 'КМ',
 'КН',
 'КО',
 'КП',
 'КР',
 'КС',
 'КТ',
 'КУ',
 'КФ',
 'КХ',
 'КЦ',
 'КЧ',
 'КШ',
 'КЩ',
 'КЪ',
 'КЫ',
 'КЬ',
 'КЭ',
 'КЮ',
 'КЯ',
 'ЛМ',
 'ЛН',
 'ЛО',
 'ЛП',
 'ЛР',
 'ЛС',
 'ЛТ',
 'ЛУ',
 'ЛФ',
 'ЛХ',
 'ЛЦ',
 'ЛЧ',
 'ЛШ',
 'ЛЩ',
 'ЛЪ',
 'ЛЫ',
 'ЛЬ',
 'ЛЭ',
 'ЛЮ',
 'ЛЯ',
 'МН',
 'МО',
 'МП',
 'МР',
 'МС',
 'МТ',
 'МУ',
 'МФ',
 'МХ',
 'МЦ',
 'МЧ',
 'МШ',
 'МЩ',
 'МЪ',
 'МЫ',
 'МЬ',
 'МЭ',
 'МЮ',
 'МЯ',
 'НО',
 'НП',
 'НР',
 'НС',
 'НТ',
 'НУ',
 'НФ',
 'НХ',
 'НЦ',
 'НЧ',
 'НШ',
 'НЩ',
 'НЪ',
 'НЫ',
 'НЬ',
 'НЭ',
 'НЮ',
 'НЯ',
 'ОП',
 'ОР',
 'ОС',
 'ОТ',
 'ОУ',
 'ОФ',
 'ОХ',
 'ОЦ',
 'ОЧ',
 'ОШ',
 'ОЩ',
 'ОЪ',
 'ОЫ',
 'ОЬ',
 'ОЭ',
 'ОЮ',
 'ОЯ',
 'ПР',
 'ПС',
 'ПТ',
 'ПУ',
 'ПФ',
 'ПХ',
 'ПЦ',
 'ПЧ',
 'ПШ',
 'ПЩ',
 'ПЪ',
 'ПЫ',
 'ПЬ',
 'ПЭ',
 'ПЮ',
 'ПЯ',
 'РС',
 'РТ',
 'РУ',
 'РФ',
 'РХ',
 'РЦ',
 'РЧ',
 'РШ',
 'РЩ',
 'РЪ',
 'РЫ',
 'РЬ',
 'РЭ',
 'РЮ',
 'РЯ',
 'СТ',
 'СУ',
 'СФ',
 'СХ',
 'СЦ',
 'СЧ',
 'СШ',
 'СЩ',
 'СЪ',
 'СЫ',
 'СЬ',
 'СЭ',
 'СЮ',
 'СЯ',
 'ТУ',
 'ТФ',
 'ТХ',
 'ТЦ',
 'ТЧ',
 'ТШ',
 'ТЩ',
 'ТЪ',
 'ТЫ',
 'ТЬ',
 'ТЭ',
 'ТЮ',
 'ТЯ',
 'УФ',
 'УХ',
 'УЦ',
 'УЧ',
 'УШ',
 'УЩ',
 'УЪ',
 'УЫ',
 'УЬ',
 'УЭ',
 'УЮ',
 'УЯ',
 'ФХ',
 'ФЦ',
 'ФЧ',
 'ФШ',
 'ФЩ',
 'ФЪ',
 'ФЫ',
 'ФЬ',
 'ФЭ',
 'ФЮ',
 'ФЯ',
 'ХЦ',
 'ХЧ',
 'ХШ',
 'ХЩ',
 'ХЪ',
 'ХЫ',
 'ХЬ',
 'ХЭ',
 'ХЮ',
 'ХЯ',
 'ЦЧ',
 'ЦШ',
 'ЦЩ',
 'ЦЪ',
 'ЦЫ',
 'ЦЬ',
 'ЦЭ',
 'ЦЮ',
 'ЦЯ',
 'ЧШ',
 'ЧЩ',
 'ЧЪ',
 'ЧЫ',
 'ЧЬ',
 'ЧЭ',
 'ЧЮ',
 'ЧЯ',
 'ШЩ',
 'ШЪ',
 'ШЫ',
 'ШЬ',
 'ШЭ',
 'ШЮ',
 'ШЯ',
 'ЩЪ',
 'ЩЫ',
 'ЩЬ',
 'ЩЭ',
 'ЩЮ',
 'ЩЯ',
 'ЪЫ',
 'ЪЬ',
 'ЪЭ',
 'ЪЮ',
 'ЪЯ',
 'ЫЬ',
 'ЫЭ',
 'ЫЮ',
 'ЫЯ',
 'ЬЭ',
 'ЬЮ',
 'ЬЯ',
 'ЭЮ',
 'ЭЯ',
 'ЮЯ']



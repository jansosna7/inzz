import random

xml_file = '''<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2024-03-21 11:39:30 by Eclipse SUMO netedit Version 1.17.0
-->

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Routes -->
    <route id="r_0" edges="A7B7 B7C7 C7D7 D7E7"/>
    <route id="r_1" edges="F7E7 E7D7 D7C7 C7B7"/>
    <route id="r_10" edges="H5H6 H6H7 H7H8"/>
    <route id="r_11" edges="H9H8 H8H7 H7H6"/>
    <route id="r_12" edges="A4B4 B4C4 C4D4 D4E4"/>
    <route id="r_13" edges="F4E4 E4D4 D4C4 C4B4"/>
    <route id="r_14" edges="C2C3 C3C4 C4C5"/>
    <route id="r_15" edges="C6C5 C5C4 C4C3"/>
    <route id="r_16" edges="D2D3 D3D4 D4D5"/>
    <route id="r_17" edges="D6D5 D5D4 D4D3"/>
    <route id="r_18" edges="E4F4 F4G4 G4H4 H4I4"/>
    <route id="r_19" edges="J4I4 I4H4 H4G4 G4F4"/>
    <route id="r_2" edges="C5C6 C6C7 C7C8"/>
    <route id="r_20" edges="G2G3 G3G4 G4G5"/>
    <route id="r_21" edges="G6G5 G5G4 G4G3"/>
    <route id="r_22" edges="H2H3 H3H4 H4H5"/>
    <route id="r_23" edges="H6H5 H5H4 H4H3"/>
    <route id="r_3" edges="C9C8 C8C7 C7C6"/>
    <route id="r_4" edges="D5D6 D6D7 D7D8"/>
    <route id="r_5" edges="D9D8 D8D7 D7D6"/>
    <route id="r_6" edges="E7F7 F7G7 G7H7 H7I7"/>
    <route id="r_7" edges="J7I7 I7H7 H7G7 G7F7"/>
    <route id="r_8" edges="G5G6 G6G7 G7G8"/>
    <route id="r_9" edges="G9G8 G8G7 G7G6"/>
	
'''
        #id  beg  end pro rou
rows = [[0  ,0   ,15  ,0.3  ,0  ],
        [0  ,0   ,15  ,0.1  ,1  ],
        [0  ,0   ,30  ,0.1  ,2  ],
        [0  ,0   ,30  ,0.1  ,3  ],
        [0  ,0   ,30  ,0.2  ,4  ],
        [0  ,0   ,30  ,0.1  ,5  ],

        [1  ,15  ,30  ,0.1  ,0  ],
        [1  ,15  ,30  ,0.3  ,1  ],

        [2  ,35  ,65  ,0.1  ,0  ],
        [2  ,35  ,65  ,0.1  ,1  ],
        [2  ,35  ,50  ,0.3  ,2  ],
        [2  ,35  ,50  ,0.1  ,3  ],
        [2  ,35  ,65  ,0.1  ,4  ],
        [2  ,35  ,65  ,0.2  ,5  ],

        [3  ,50  ,65  ,0.1  ,2  ],
        [3  ,50  ,65  ,0.3  ,3  ],

        [4  ,70  ,100 ,0.2  ,0  ],
        [4  ,70  ,100 ,0.1  ,1  ],
        [4  ,70  ,100 ,0.1  ,2  ],
        [4  ,70  ,100 ,0.1  ,3  ],
        [4  ,70  ,85  ,0.3  ,4  ],
        [4  ,70  ,85  ,0.1  ,5  ],

        [5  ,85  ,100 ,0.1  ,4  ],
        [5  ,85  ,100 ,0.3  ,5  ],


        [0  ,0   ,100 ,0.18 ,6  ],
        [0  ,0   ,100 ,0.18 ,7  ],
        [0  ,0   ,100 ,0.18 ,8  ],
        [0  ,0   ,100 ,0.18 ,9  ],
        [0  ,0   ,100 ,0.18 ,10 ],
        [0  ,0   ,100 ,0.18 ,11 ],


        [0  ,0   ,25  ,0.34 ,12 ],
        [0  ,0   ,25  ,0.1  ,13 ],
        [0  ,0   ,25  ,0.33 ,14 ],
        [0  ,0   ,25  ,0.1  ,15 ],
        [0  ,0   ,25  ,0.32 ,16 ],
        [0  ,0   ,25  ,0.1  ,17 ],

        [1  ,25  ,50  ,0.1  ,12 ],
        [1  ,25  ,50  ,0.32 ,13 ],
        [1  ,25  ,50  ,0.1  ,14 ],
        [1  ,25  ,50  ,0.34 ,15 ],
        [1  ,25  ,50  ,0.1  ,16 ],
        [1  ,25  ,50  ,0.33 ,17 ],

        [2  ,50  ,75  ,0.33 ,12 ],
        [2  ,50  ,75  ,0.32 ,13 ],
        [2  ,50  ,75  ,0.34 ,14 ],
        [2  ,50  ,75  ,0.1  ,15 ],
        [2  ,50  ,75  ,0.1  ,16 ],
        [2  ,50  ,75  ,0.1  ,17 ],

        [3  ,75  ,100 ,0.1  ,12 ],
        [3  ,75  ,100 ,0.1  ,13 ],
        [3  ,75  ,100 ,0.32 ,14 ],
        [3  ,75  ,100 ,0.34 ,15 ],
        [3  ,75  ,100 ,0.1  ,16 ],
        [3  ,75  ,100 ,0.3  ,17 ],
        ]




sorted_array = sorted(rows, key=lambda x: x[1])

file_path = "double_2.rou.xml"
with open(file_path, 'w') as file:
    file.write(xml_file)

    for r in sorted_array:
        line = '''   <flow id="f_'''+str(r[0])+"_"+str(r[4])+'''" begin="'''+str(r[1])+'''.00" end="'''+str(r[2])+'''.00" probability="'''+str(r[3])+'''" route="r_'''+str(r[4])+'''"/>'''
        file.write(line)
        file.write("\n")
        
    file.write("</routes>")

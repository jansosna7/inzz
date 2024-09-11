import random

xml_file = '''<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2024-07-16 18:36:14 by Eclipse SUMO netedit Version 1.17.0
-->

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Routes -->
    <route id="r_0" edges="A5B5 B5C5 C5D5 D5E5 E5F5"/>
    <route id="r_1" edges="G5F5 F5E5 E5D5 D5C5 C5B5"/>
    <route id="r_10" edges="C4C3 C3C2 C2C1"/>
    <route id="r_11" edges="C0C1 C1C2 C2C3"/>
    <route id="r_12" edges="D4D3 D3D2 D2D1"/>
    <route id="r_13" edges="D0D1 D1D2 D2D3"/>
    <route id="r_14" edges="E4E3 E3E2 E2E1"/>
    <route id="r_15" edges="E0E1 E1E2 E2E3"/>
    <route id="r_2" edges="C7C6 C6C5 C5C4"/>
    <route id="r_3" edges="C3C4 C4C5 C5C6"/>
    <route id="r_4" edges="D7D6 D6D5 D5D4"/>
    <route id="r_5" edges="D3D4 D4D5 D5D6"/>
    <route id="r_6" edges="E7E6 E6E5 E5E4"/>
    <route id="r_7" edges="E3E4 E4E5 E5E6"/>
    <route id="r_8" edges="A2B2 B2C2 C2D2 D2E2 E2F2"/>
    <route id="r_9" edges="G2F2 F2E2 E2D2 D2C2 C2B2"/>
	
'''

        #id  beg  end pro rou
rows = [[0  ,0   ,100 ,0.3  ,0  ],
        [0  ,0   ,100 ,0.3  ,1  ],

        [0  ,0   ,50 ,0.3  ,10 ],
        [0  ,0   ,50 ,0.3  ,11 ],
        [0  ,20  ,70 ,0.3  ,12 ],
        [0  ,20  ,70 ,0.3  ,13 ],
        [0  ,40  ,100,0.3  ,14 ],
        [0  ,40  ,100,0.3  ,15 ],
        [1  ,80  ,100,0.3  ,10 ],
        [1  ,80  ,100,0.3  ,11 ]
        ]

sorted_array = sorted(rows, key=lambda x: x[1])

file_path = "triple_1.rou.xml"
with open(file_path, 'w') as file:
    file.write(xml_file)

    for r in sorted_array:
        line = '''   <flow id="f_'''+str(r[0])+"_"+str(r[4])+'''" begin="'''+str(r[1])+'''.00" end="'''+str(r[2])+'''.00" probability="'''+str(r[3])+'''" route="r_'''+str(r[4])+'''"/>'''
        file.write(line)
        file.write("\n")
        
    file.write("</routes>")

import random

xml_file = '''<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2024-01-16 22:44:19 by Eclipse SUMO netedit Version 1.17.0
-->

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Routes -->
    <route id="r_0" edges="A2B2 B2C2 C2D2"/>
    <route id="r_1" edges="C4C3 C3C2 C2C1"/>
    <route id="r_2" edges="E2D2 D2C2 C2B2"/>
    <route id="r_3" edges="C0C1 C1C2 C2C3"/>
	
'''

        #id  beg  end pro rou
rows = [[0  ,0   ,30  ,0.21 ,0  ],
        [0  ,40  ,70  ,0.21 ,1  ],
        [0  ,80  ,110 ,0.21 ,2  ],
        [0  ,120 ,150 ,0.21 ,3  ]
        ]

for rou in range(4):
    nr = 1
    for begin in range(200,1200,50):
        mid = 0.13
        prob = mid + (nr/200) + random.uniform(-mid, mid)
        rows.append([nr, begin, begin+47, prob, rou])
        nr = nr + 1

sorted_array = sorted(rows, key=lambda x: x[1])

file_path = "single_5.rou.xml"
with open(file_path, 'w') as file:
    file.write(xml_file)

    for r in sorted_array:
        line = '''   <flow id="f_'''+str(r[0])+"_"+str(r[4])+'''" begin="'''+str(r[1])+'''.00" end="'''+str(r[2])+'''.00" probability="'''+str(r[3])+'''" route="r_'''+str(r[4])+'''"/>'''
        file.write(line)
        file.write("\n")
        
    file.write("</routes>")

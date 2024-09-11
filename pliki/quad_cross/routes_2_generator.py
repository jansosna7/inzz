import random

xml_file = '''<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 2024-08-16 20:35:14 by Eclipse SUMO netedit Version 1.17.0
-->

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Routes -->
    <route id="r_0" edges="A9B9 B9C9 C9D9 D9E9"/>
    <route id="r_1" edges="F9E9 E9D9 D9C9 C9B9"/>
    <route id="r_10" edges="E8F8 F8G8 G8H8 H8I8"/>
    <route id="r_11" edges="J8I8 I8H8 H8G8 G8F8"/>
    <route id="r_12" edges="G11G10 G10G9 G9G8 G8G7"/>
    <route id="r_13" edges="G6G7 G7G8 G8G9 G9G10"/>
    <route id="r_14" edges="H11H10 H10H9 H9H8 H8H7"/>
    <route id="r_15" edges="H6H7 H7H8 H8H9 H9H10"/>
    <route id="r_16" edges="A5B5 B5C5 C5D5 D5E5"/>
    <route id="r_17" edges="F5E5 E5D5 D5C5 C5B5"/>
    <route id="r_18" edges="A4B4 B4C4 C4D4 D4E4"/>
    <route id="r_19" edges="F4E4 E4D4 D4C4 C4B4"/>
    <route id="r_2" edges="A8B8 B8C8 C8D8 D8E8"/>
    <route id="r_20" edges="C7C6 C6C5 C5C4 C4C3"/>
    <route id="r_21" edges="C2C3 C3C4 C4C5 C5C6"/>
    <route id="r_22" edges="D7D6 D6D5 D5D4 D4D3"/>
    <route id="r_23" edges="D2D3 D3D4 D4D5 D5D6"/>
    <route id="r_24" edges="E5F5 F5G5 G5H5 H5I5"/>
    <route id="r_25" edges="J5I5 I5H5 H5G5 G5F5"/>
    <route id="r_26" edges="E4F4 F4G4 G4H4 H4I4"/>
    <route id="r_27" edges="J4I4 I4H4 H4G4 G4F4"/>
    <route id="r_28" edges="G7G6 G6G5 G5G4 G4G3"/>
    <route id="r_29" edges="G2G3 G3G4 G4G5 G5G6"/>
    <route id="r_3" edges="F8E8 E8D8 D8C8 C8B8"/>
    <route id="r_30" edges="H7H6 H6H5 H5H4 H4H3"/>
    <route id="r_31" edges="H2H3 H3H4 H4H5 H5H6"/>
    <route id="r_4" edges="C11C10 C10C9 C9C8 C8C7"/>
    <route id="r_5" edges="C6C7 C7C8 C8C9 C9C10"/>
    <route id="r_6" edges="D11D10 D10D9 D9D8 D8D7"/>
    <route id="r_7" edges="D6D7 D7D8 D8D9 D9D10"/>
    <route id="r_8" edges="E9F9 F9G9 G9H9 H9I9"/>
    <route id="r_9" edges="J9I9 I9H9 H9G9 G9F9"/>
	
'''

        #id  beg  end pro rou
rows = [[0  ,0   ,50 ,0.3  ,0  ],
        [0  ,0   ,50 ,0.3  ,1  ],
        [0  ,50  ,100,0.3  ,2  ],
        [0  ,50  ,100,0.3  ,3  ],
        [0  ,0   ,50 ,0.3  ,4  ],
        [0  ,0   ,50 ,0.3  ,5  ],
        [0  ,50  ,100,0.3  ,6  ],
        [0  ,50  ,100,0.3  ,7  ],

        [0  ,0   ,200,0.25 ,8  ],
        [0  ,0   ,200,0.25 ,9  ],
        [0  ,0   ,200,0.25 ,10 ],
        [0  ,0   ,200,0.25 ,11 ],
        [0  ,0   ,200,0.25 ,12 ],
        [0  ,0   ,200,0.25 ,13 ],
        [0  ,0   ,200,0.25 ,14 ],
        [0  ,0   ,200,0.25 ,15 ],
        
        [0  ,0   ,80  ,0.3  ,16 ],
        [0  ,20  ,100 ,0.3  ,17 ],
        [0  ,0   ,80  ,0.3  ,18 ],
        [0  ,20  ,100 ,0.3  ,19 ],
        [0  ,0   ,70  ,0.3  ,20 ],
        [0  ,30  ,100 ,0.3  ,21 ],
        [0  ,30  ,100 ,0.3  ,22 ],
        [0  ,0   ,70  ,0.3  ,23 ],        
        [1  ,100 ,150 ,0.3  ,16 ],
        [1  ,110 ,160 ,0.3  ,17 ],
        [1  ,120 ,170 ,0.3  ,18 ],
        [1  ,130 ,180 ,0.3  ,19 ],
        [1  ,140 ,190 ,0.3  ,20 ],
        [1  ,150 ,200 ,0.3  ,21 ],
        [1  ,160 ,210 ,0.3  ,22 ],
        [1  ,170 ,220 ,0.3  ,23 ]
        ]

for rou in range(24,32):
    nr = 1
    for begin in range(0,230,33):
        mid = 0.2
        prob = mid + random.uniform(-mid, mid)
        rows.append([nr, begin, begin+30, prob, rou])
        nr = nr + 1

sorted_array = sorted(rows, key=lambda x: x[1])

file_path = "quad_2.rou.xml"
with open(file_path, 'w') as file:
    file.write(xml_file)

    for r in sorted_array:
        line = '''   <flow id="f_'''+str(r[0])+"_"+str(r[4])+'''" begin="'''+str(r[1])+'''.00" end="'''+str(r[2])+'''.00" probability="'''+str(r[3])+'''" route="r_'''+str(r[4])+'''"/>'''
        file.write(line)
        file.write("\n")
        
    file.write("</routes>")

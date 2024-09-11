import xml.etree.ElementTree as ET

def double_probabilities(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    for flow in root.findall('flow'):
        probability = float(flow.get('probability'))
        new_probability = probability * 2
        
        flow.set('probability', str(new_probability))

    tree.write(output_file)

for i in range(5):
    double_probabilities("single_project_"+str(i+1)+".rou.xml", "project_"+str(i+1)+".rou.xml")

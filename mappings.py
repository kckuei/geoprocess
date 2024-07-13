from collections import defaultdict


# Define the sbt mapping dictionary
sbt_mapping = {
    'Sensitive fine grained': 1,
    'Organic soils - clay': 2,
    'Clays: clay to silty clay': 3,
    'Silt mixtures: clayey silt and silty clay': 4,
    'Sand mixtures: silty sand to sandy silt': 5,
    'Sands: clean sands to silty sands': 6,
    'Dense sand to gravelly sand': 7,
    'Very stiff sand to clayey sand*': 8,
    'Very stiff fine-grained*': 9
}


# Define the sbt color mapping
color_mapping = {
    1: '#D3291C',
    2: '#B36A3F',
    3: '#4A5777',
    4: '#479085',
    5: '#7EC4A0',
    6: '#BDA464',
    7: '#EB9D4A',
    8: '#999999',
    9: '#DEDEDE'
}


# Define the color mapping for distinct layers
color_mapping_layers = defaultdict(
    lambda: "000000", 
    {'Embankment Zone C': '#ed876f',
     'Embankment Zone D': '#ed876f',
     'Embankment Zone F': '#ed876f',
     'Alluvium': '#fadd7f'
    })

if __name__ == '__main__':
    
    None
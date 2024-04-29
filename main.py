import sorobn as hh
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import networkx as nx
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import streamlit as st

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define the structure of the Bayesian Network
# The structure is defined as a list of tuples, where each tuple represents a directed edge between two nodes
model = BayesianNetwork([
    ('wind', 'wind_wave_effect'),
    ('wave', 'wind_wave_effect'),
    ('wind', 'wave'),
    ('wind_wave_effect', 'ice_thickness'),
    ('wind_wave_effect', 'ice_concentration'),
    ('ice_thickness', 'ship_speed'),
    ('ice_thickness', 'getting_stuck_in_the_ice'),
    ('ice_concentration', 'ship_speed'),
    ('ice_concentration', 'ship-ice_collision'),
    ('ship_speed', 'getting_stuck_in_the_ice'),
    ('ship_speed', 'ship-ice_collision'),
    ('getting_stuck_in_the_ice', 'composite_risk'),
    ('ship-ice_collision', 'composite_risk')
])

# Define the states for each node
# With the help of these states, we can define the Conditional Probability Tables (CPTs) for each node
states = {
    'ice_concentration': ['<50%', '50-70%', '>70%'],
    'ice_thickness': ['<40', '40-80', '>80'],
    'ship_speed': ['<5', '5-8', '8-11', '>11'],
    'wave': ['<0.5', '0.5-1.25', '>1.25'],
    'wind': ['<5.5', '5.5-7.9', '>7.9'],
    'wind_wave_effect': ['Severe', 'Low'],
    'getting_stuck_in_the_ice': ['Remote', 'Possible', 'Probable'],
    'ship-ice_collision': ['Remote', 'Possible', 'Probable'],
    'composite_risk': ['Low', 'Medium', 'High']
}


# since we are storing the values in excel, we need to extract them
def extract_values(cpt):
    """
    Handle extracting values from a pandas DataFrame, some rows might be first 3 rows of just titles.
    WE need extract values in row so [row,row,row], values will only be digits of 0-1 in format X.X.
    """
    values = []
    for row in cpt.iterrows():
        # all after first col in row
        val = row[1][1:]
        val = [str(v) for v in val]
        # try match all X.X format, check that all values are in X.X format can alos be just 0
        if not all(re.match(r'\d\.\d|0', v) for v in val):
            continue
        values.append([float(v) for v in val])
    return values


SOURCE = "cpts.xlsx" # Path to the Excel file containing the CPTs

# Define the CPT for 'wind' with the probabilities for wind being in the three states that we hard coded
cpd_wind = TabularCPD(variable='wind', variable_card=3,
                        values=[[0.5], [0.3], [0.2]],
                        state_names={'wind': states['wind']}
                      )  # Probabilities for wind being in the three states




# we replicate the same pattern for all the other nodes
# 1. Load the data from the Excel file
# 2. Extract the values from the data
# 3. Define the CPT for the node: (i) provide the variable name, (ii) the number of states, (iii) the values, and (iv) the state names for each variable
# 4. Add the CPT to the model
# 5. Repeat for all nodes


#==========
# WAVE NODE
#==========


cpt = pd.read_excel(SOURCE, sheet_name='wave')
wave_values = extract_values(cpt)

cpd_wave = TabularCPD(variable='wave', variable_card=3,
                        evidence=['wind'],
                        evidence_card=[3],
                        values=wave_values,
                        state_names={'wind': states['wind'],
                                     'wave': states['wave']}
                    )


#==========
# WIND WAVE EFFECT NODE
#==========



wind_wave_effect = pd.read_excel(SOURCE, sheet_name='wind_wave_effect')
wind_wave_effect_values = extract_values(wind_wave_effect)

# Define the CPT for 'wind_wave_effect', depending on 'wind' and 'wave'
cpd_wind_wave_effect = TabularCPD(variable='wind_wave_effect', variable_card=2,
                                    values=wind_wave_effect_values,
                                    evidence=['wind', 'wave'],
                                    evidence_card=[3, 3],
                                    state_names={'wind': states['wind'],
                                                 'wave': states['wave'],
                                                 'wind_wave_effect': states['wind_wave_effect']}
                                  )


#==========
# ICE THICKNESS NODE
#==========

ice_thickness = pd.read_excel(SOURCE, sheet_name='ice_thickness')
ice_thickness_values = extract_values(ice_thickness)

# Define the CPT for 'ice_thickness', depending on 'wind_wave_effect'
cpd_ice_thickness = TabularCPD(variable='ice_thickness', variable_card=3,
                                values=ice_thickness_values,
                                evidence=['wind_wave_effect'],
                                evidence_card=[2],
                                state_names={'wind_wave_effect': states['wind_wave_effect'],
                                            'ice_thickness': states['ice_thickness']}
                                )


#==========
# ICE CONCENTRATION NODE
#==========

ice_concentration = pd.read_excel(SOURCE, sheet_name='ice_concentration')
ice_concentration_values = extract_values(ice_concentration)

# Define the CPT for 'ice_concentration', depending on 'wind_wave_effect'
cpd_ice_concentration = TabularCPD(variable='ice_concentration', variable_card=3,
                                    values=ice_concentration_values,
                                    evidence=['wind_wave_effect'],
                                    evidence_card=[2],
                                    state_names={'wind_wave_effect': states['wind_wave_effect'],
                                                'ice_concentration': states['ice_concentration']}
                                    )


#==========
# SHIP SPEED NODE
#==========


ship_speed = pd.read_excel(SOURCE, sheet_name='ship_speed')
ship_speed_values = extract_values(ship_speed)

# Define the CPT for 'ship_speed', depending on 'ice_thickness' and 'ice_concentration'
cpd_ship_speed = TabularCPD(variable='ship_speed', variable_card=4,
                                values=ship_speed_values,
                                evidence=['ice_thickness', 'ice_concentration'],
                                evidence_card=[3, 3],
                                state_names={'ice_thickness': states['ice_thickness'],
                                            'ice_concentration': states['ice_concentration'],
                                            'ship_speed': states['ship_speed']}
                                )

#==========
# GETTING STUCK IN THE ICE NODE
#==========


getting_stuck = pd.read_excel(SOURCE, sheet_name='getting_stuck')
getting_stuck_values = extract_values(getting_stuck)

# Define the CPT for 'getting_stuck_in_the_ice', depending on 'ship_speed'
cpd_getting_stuck = TabularCPD(variable='getting_stuck_in_the_ice', variable_card=3,
                                values=getting_stuck_values,
                                evidence=['ship_speed', 'ice_thickness'],
                                evidence_card=[4, 3],
                                state_names={'ship_speed': states['ship_speed'],
                                            'ice_thickness': states['ice_thickness'],
                                            'getting_stuck_in_the_ice': states['getting_stuck_in_the_ice']}
                                 )

#==========
# SHIP-ICE COLLISION NODE
#==========

ship_ice_collision = pd.read_excel(SOURCE, sheet_name='ship_ice_collision')
ship_ice_collision_values = extract_values(ship_ice_collision)


# Define the CPT for 'ship-ice_collision', depending on 'ship_speed'
cpd_ship_ice_collision = TabularCPD(variable='ship-ice_collision', variable_card=3,
                                    values=ship_ice_collision_values,
                                    evidence=['ship_speed', 'ice_concentration'],
                                    evidence_card=[4,3],
                                    state_names={'ship_speed': states['ship_speed'],
                                                'ice_concentration': states['ice_concentration'],
                                                'ship-ice_collision': states['ship-ice_collision']}
                                    )

#==========
# COMPOSITE RISK NODE
#==========


composite_risk = pd.read_excel(SOURCE, sheet_name='composite_risk')
composite_risk_values = extract_values(composite_risk)

# Define the CPT for 'composite_risk', depending on 'getting_stuck_in_the_ice' and 'ship-ice_collision'
cpd_composite_risk = TabularCPD(variable='composite_risk', variable_card=3,
                                values=composite_risk_values,
                                evidence=['getting_stuck_in_the_ice', 'ship-ice_collision'],
                                evidence_card=[3, 3],
                                state_names={'getting_stuck_in_the_ice': states['getting_stuck_in_the_ice'],
                                            'ship-ice_collision': states['ship-ice_collision'],
                                            'composite_risk': states['composite_risk']}
                                )



# Now we add all the CPTs to the model
cpts = [cpd_wind,
        cpd_wave,
        cpd_wind_wave_effect,
        cpd_ice_thickness,
        cpd_ice_concentration,
        cpd_ship_speed,
        cpd_getting_stuck,
        cpd_ship_ice_collision,
        cpd_composite_risk]

model.add_cpds(*cpts)



# Validate the model to ensure the CPTs are correctly defined and compatible
assert model.check_model()


# Create the inference object in order to perform queries
inference = VariableElimination(model)

# Define the graph for the Bayesian Network
G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())

@st.cache_resource # we cache the graph becauase we dont want to recompute it every time
def get_gr():
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight='bold')
    plt.title('Bayesian Network Graph')
    plt.savefig('graph.png')

    # probably a neater way to do this but this works
    st.title('Bayesian Network')

    st.write('This is a simple example of a Bayesian Network. The nodes represent different events and the edges represent the dependencies between them.')

    return 'graph.png'

st.image(get_gr())



# define user inputs for the network
wind = st.selectbox('Wind', states['wind'])
wave = st.selectbox('Wave', states['wave'])

# Now we computer the composite risk based on the user inputs
result = inference.query(
    variables=['composite_risk'],
    evidence={'wind': wind, 'wave': wave}
) # P(composite_risk | wind, wave)
max_index = np.argmax(result.values) # get the index of the maximum value
mxvar = result.state_names['composite_risk'][max_index] # get the state name of the maximum value
st.write(f'Composite Risk: {mxvar}')

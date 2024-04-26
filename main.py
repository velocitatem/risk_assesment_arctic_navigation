import sorobn as hh
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import streamlit as st

with open('edges.json') as f:
    edges = json.load(f)
edges = [tuple([name.replace(' ', '_') for name in edge]) for edge in edges]
bn = hh.BayesNet(*edges)

STATES = {
    'ice_concentration': ['<50%', '50–70%', '> 70%'],
    'ice_thickness': ['<40', '40–80', '>80'],
    'ship_speed': ['<5', '5–8', '>8'],
    'wave': ['<0.5', '0.5–1.25', '> 1.25'],
    'wind': ['<5.5', '5.5–7.9', '> 7.9'],
    'wind_wave_effect': ['Low', 'Possible', 'Probable', 'Critical'],
    'getting_stuck_in_the_ice': ['Severe', 'Remote', 'Minor'],
    'ship-ice_collision': ['Remote', 'Possible', 'Probable', 'Critical'],
    'consequences_of_getting_stuck_in_the_ice': ['Minor', 'Major'],
    'ship_ice_collision_consequences': ['Major', 'Critical'],
    'risk_of_getting_stuck_in_the_ice': ['Low', 'Medium', 'High'],
    'ship_ice_collision_risk': ['Low', 'Medium', 'High'],
    'composite_risk': ['Low', 'Medium', 'High']
}


@st.cache_resource
def get_samples():
    samples = []
    for i in range(500):
        sample = {key: np.random.choice(STATES[key]) for key in STATES.keys()}
        samples.append(sample)
    return samples

samples = get_samples()
bn.fit(pd.DataFrame(samples))



print(samples)

bn.prepare()


@st.cache_resource
def get_plot():
    dot = bn.graphviz()
    path = dot.render('graph', format='png', cleanup=True)
    return path

st.title('Bayesian Network')

st.write('This is a simple example of a Bayesian Network. The nodes represent different events and the edges represent the dependencies between them.')

st.image(get_plot())

# let user input causes wind, wave
wind = st.selectbox('Wind speed', STATES['wind'])
wave = st.selectbox('Wave height', STATES['wave'])

# calculate the probability of getting stuck in the ice
p = bn.query('composite_risk', event={'wind': wind, 'wave': wave})
# get argmax
risk = p.idxmax()
st.write(f'## The risk of getting stuck in the ice is {risk}')



st.write('The nodes and states distribution are as follows:')
# for each node, show the states
for node in bn.nodes:
    st.write(f'## {node}')
    st.write(bn.P[node])

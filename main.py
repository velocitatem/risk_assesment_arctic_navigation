import sorobn as hh
import pandas as pd
from matplotlib import pyplot as plt
import json
import streamlit as st

with open('edges.json') as f:
    edges = json.load(f)
edges = [tuple([name.replace(' ', '_') for name in edge]) for edge in edges]
bn = hh.BayesNet(*edges)
bn.prepare()

@st.cache_resource
def get_plot():
    dot = bn.graphviz()
    path = dot.render('graph', format='png', cleanup=True)
    return path

st.title('Bayesian Network')

st.write('This is a simple example of a Bayesian Network. The nodes represent different events and the edges represent the dependencies between them.')

st.image(get_plot())


bn.sample(1000)

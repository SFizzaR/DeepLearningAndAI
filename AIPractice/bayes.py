from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
model = DiscreteBayesianNetwork([
    ('disease', 'fever'),
    ('disease', 'cough'),
    ('disease', 'fatigue'),
    ('disease', 'chills')
])

# CPD for disease (prior)
cpd_disease = TabularCPD(
    variable='disease',
    variable_card=2,
    values=[[0.4], [0.6]],
    state_names={'disease': ['flu', 'cold']}
)

# CPD for fever given disease
cpd_fever = TabularCPD(
    variable='fever',
    variable_card=2,
    evidence=['disease'],
    evidence_card=[2],
    values=[[0.8, 0.2],  # P(fever=no | flu), P(fever=no | cold)
            [0.2, 0.8]], # P(fever=yes | flu), P(fever=yes | cold)
    state_names={
        'fever': ['no', 'yes'],
        'disease': ['flu', 'cold']
    }
)

# CPD for cough given disease
cpd_cough = TabularCPD(
    variable='cough',
    variable_card=2,
    evidence=['disease'],
    evidence_card=[2],
    values=[[0.7, 0.3],  # P(cough=no | flu), P(cough=no | cold)
            [0.3, 0.7]],
    state_names={
        'cough': ['no', 'yes'],
        'disease': ['flu', 'cold']
    }
)

# CPD for fatigue given disease
cpd_fatigue = TabularCPD(
    variable='fatigue',
    variable_card=2,
    evidence=['disease'],
    evidence_card=[2],
    values=[[0.6, 0.4],
            [0.4, 0.6]],
    state_names={
        'fatigue': ['no', 'yes'],
        'disease': ['flu', 'cold']
    }
)

# CPD for chills given disease
cpd_chills = TabularCPD(
    variable='chills',
    variable_card=2,
    evidence=['disease'],
    evidence_card=[2],
    values=[[0.7, 0.5],
            [0.3, 0.5]],
    state_names={
        'chills': ['no', 'yes'],
        'disease': ['flu', 'cold']
    }
)

# Add CPDs to model
model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)

# Check the model
assert model.check_model()

# Inference
infer = VariableElimination(model)

# Query: What is the probability of disease given fever=yes and cough=yes?
result = infer.query(
    variables=['disease'],
    evidence={'fever': 'yes', 'cough': 'yes'}
)

print("Probability of disease given fever and cough:\n", result)

# Extra: probability of fatigue given disease = flu
result_fatigue = infer.query(
    variables=['fatigue'],
    evidence={'disease': 'flu'}
)

print("\nProbability of fatigue given disease=flu:\n", result_fatigue)

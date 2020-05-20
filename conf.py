import logging as log
from rdflib.namespace import Namespace

ScholalryData = Namespace('https://w3id.org/scholarlydata/ontology/conference-ontology.owl#')
IOnt = Namespace('https://w3id.org/scholarlydata/ontology/indicators-ontology.owl#')
AltmetricsCOVID = Namespace('http://w3id.org/stlab/altmetrics_covid-19/')

INSTTOKEN = ""
APIKEY = ""
LOG_LEVEL = log.INFO
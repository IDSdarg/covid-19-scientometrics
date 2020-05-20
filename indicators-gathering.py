'''
Created on May 8, 2020

@author: Andrea Giovanni Nuzzolese
'''
from argparse import ArgumentParser
import codecs
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Dict
import urllib.request, urllib.parse

from pandas import DataFrame
from rdflib.graph import Graph
from rdflib.namespace import RDF, RDFS, Namespace, XSD, DC
from rdflib.term import URIRef, Literal

from conf import ScholalryData, IOnt, AltmetricsCOVID
import conf
import logging as log
import pandas as pd
import datetime

class Paper():
    
    def __init__(self, doi, date_time, timestamp):
        self.__doi = doi
        self.__date_time = date_time
        self.__timestamp = timestamp
        
    def get_doi(self):
        return self.__doi
    
    def get_date_time(self):
        return self.__date_time
    
    def get_timestamp(self):
        return self.__timestamp


class CrossRefAPIClient():
    
    CROSS_REF_ENDPOINT= "https://api.crossref.org/works"
    
    def retrieve_doi(self, first_author: str, year: str, title: str) -> Dict:
        
        dict = {}
        
        query = "%s (%s). %s" % (first_author, year, title)
        params = {'sort': 'score',
                  'query': query,
                  'rows': 1}
        
        params = urllib.parse.urlencode(params)
        
        endpoint = CrossRefAPIClient.CROSS_REF_ENDPOINT + "?%s"
        
        request = urllib.request.Request(endpoint % params)
        response = urllib.request.urlopen(request)
        
        try:
            
            
            output = response.read()
            if output is not None:
                js = json.loads(output)
                if "message" in js:
                    message = js["message"]
                    
                    if "items" in message:
                        items = message["items"]
                        
                        if len(items) > 0 and "DOI" in items[0]:
                            doi = items[0]["DOI"]
                            dict.update({"DOI": doi})
                            if "created" in items[0]:
                                created = items[0]["created"]
                                
                                if "date-time" in created:
                                    dict.update({"date-time": created["date-time"]})
                                    
                                if "timestamp" in created:
                                    dict.update({"timestamp": created["timestamp"]})
                
        except:
            log.error("No DOI found.")
        
        return dict
    
    
    def retrieve_metadata(self, doi):
        dict = {}
        
        endpoint = CrossRefAPIClient.CROSS_REF_ENDPOINT + "/%s"
        
        request = urllib.request.Request(endpoint % doi)
        
        
        try:
            
            response = urllib.request.urlopen(request)
            output = response.read()
            if output is not None:
                js = json.loads(output)
                
                if "message" in js:
                    message = js["message"]
                    
                    title = ""
                    author = ""
                    year = ""
                    
                    if "title" in message:
                        title = message["title"][0]
                    
                    if "author" in message:
                        author = "%s %s"%(message["author"][0]["given"], message["author"][0]["family"])
                    
                    if "created" in message and "date-time" in message["created"]:
                            date_time_obj = datetime.datetime.strptime(message["created"]["date-time"], '%Y-%m-%dT%H:%M:%SZ')
                            year = str(date_time_obj.year)
                            
                    dict.update({"AUTHORS": author, "YEAR": year, "TITLE": title})
                
        except:
            log.error("No metadata found for paper %s"%doi)
            traceback.print_exc()
        
        return dict


class ScopusAPIClient():
    
    CITATION_COUNT_ENDPOINT= "https://api.elsevier.com/content/abstract/citation-count"
    PLUMX_ENDPOINT= "https://api.elsevier.com/analytics/plumx/doi"
    
    def __init__(self, insttoken, api_key):
        self.__api_key = api_key
        self.__insttoken = insttoken
        
    def citation_count(self, paper: Paper) -> Graph:
        
        g = Graph()
        g.bind("sd", ScholalryData)
        g.bind("iont", IOnt)
        g.bind("covid", AltmetricsCOVID)
        
        doi = paper.get_doi()
        timestamp = paper.get_timestamp()
        
        paper = URIRef("https://doi.org/" + doi)
        
        g.add((paper, RDF.type, ScholalryData.Document))
        g.add((paper, DC.created, Literal(timestamp)))
        
        headers = { 'X-ELS-Insttoken' : self.__insttoken,
                   'X-ELS-APIKey' : self.__api_key }

        
        endpoint = ScopusAPIClient.CITATION_COUNT_ENDPOINT + "?%s"
        params = {'doi': doi}
        
        params = urllib.parse.urlencode(params)
        
        request = urllib.request.Request(endpoint % params, headers=headers)
        
        try:
            response = urllib.request.urlopen(request)
            output = response.read()
            
            if output is not None:
                js = json.loads(output)
                
                if "citation-count-response" in js:
                    ccr = js["citation-count-response"]
                    if "document" in ccr:
                        docu = ccr["document"]
                        if "citation-count" in docu:
                            
                            citation_count = docu["citation-count"]
                            
                            indicator = URIRef(AltmetricsCOVID+doi+"_citations")
                            g.add((paper, IOnt.hasIndicator, indicator))
                            g.add((indicator, RDF.type, IOnt.Indicator))
                            g.add((indicator, RDFS.label, Literal("Citations")))
                            g.add((indicator, IOnt.hasSource, AltmetricsCOVID.scopus))
                            g.add((indicator, IOnt.basedOnMetric, AltmetricsCOVID.citation_count))
                            g.add((AltmetricsCOVID.citation_count, RDF.type, IOnt.Metric))
                            g.add((indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_citations_value")))
                            g.add((URIRef(AltmetricsCOVID+doi+"_citations_value"), RDF.type, IOnt.IndicatorValue))
                            g.add((URIRef(AltmetricsCOVID+doi+"_citations_value"), IOnt.indicatorValue, Literal(citation_count, datatype=XSD.integer)))
                            
                            level_2_indicator = URIRef(AltmetricsCOVID+doi+"_citation-indexes")
                            g.add((level_2_indicator, RDF.type, IOnt.Indicator))
                            g.add((level_2_indicator, RDFS.label, Literal("Citations indexes")))
                            g.add((level_2_indicator, IOnt.hasSource, AltmetricsCOVID.scopus))
                            g.add((indicator, IOnt.hasSubIndicator, level_2_indicator))
                            g.add((level_2_indicator, IOnt.basedOnMetric, AltmetricsCOVID.citation_count))
                            g.add((AltmetricsCOVID.citation_count, RDF.type, IOnt.Metric))
                            g.add((level_2_indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_citations_value")))
                            
                            level_3_indicator = URIRef(AltmetricsCOVID+doi+"_scopus")
                            g.add((level_3_indicator, RDF.type, IOnt.Indicator))
                            g.add((level_3_indicator, RDFS.label, Literal("Scopus citation count")))
                            g.add((level_3_indicator, IOnt.hasSource, AltmetricsCOVID.scopus))
                            g.add((level_2_indicator, IOnt.hasSubIndicator, level_3_indicator))
                            g.add((level_3_indicator, IOnt.basedOnMetric, AltmetricsCOVID.citation_count))
                            g.add((AltmetricsCOVID.citation_count, RDF.type, IOnt.Metric))
                            g.add((level_3_indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_citations_value")))
        
        except:
            log.error("No citation count available for paper %s."%doi)

        return g
    
    
    def plum_x(self, paper: Paper) -> Graph:
        
        g = Graph()
        g.bind("sd", ScholalryData)
        g.bind("iont", IOnt)
        g.bind("covid", AltmetricsCOVID)
        
        doi = paper.get_doi()
        timestamp = paper.get_timestamp()
        
        paper = URIRef("https://doi.org/" + doi)
        
        g.add((paper, RDF.type, ScholalryData.Document))
        g.add((paper, DC.created, Literal(timestamp)))
        
        headers = { 'X-ELS-Insttoken' : self.__insttoken,
                   'X-ELS-APIKey' : self.__api_key }

        
        endpoint = ScopusAPIClient.PLUMX_ENDPOINT + "/%s"
        params = doi
        
        request = urllib.request.Request(endpoint % params, headers=headers)
        
        try:
            response = urllib.request.urlopen(request)
            output = response.read()
            js = json.loads(output)
        
            if output is not None:
                js = json.loads(output)
                
                if "count_categories" in js:
                    cats = js["count_categories"]
                    
                    for cat in cats:
                        
                        name = cat["name"].lower()
                        total = cat["total"]
                        
                        indicator = URIRef(AltmetricsCOVID+doi+"_"+name)
                        g.add((paper, IOnt.hasIndicator, indicator))
                        g.add((indicator, RDF.type, IOnt.Indicator))
                        g.add((indicator, RDFS.label, Literal(cat["name"])))
                        g.add((indicator, IOnt.hasSource, AltmetricsCOVID.plumx))
                        g.add((indicator, IOnt.basedOnMetric, URIRef(AltmetricsCOVID["name"])))
                        g.add((URIRef(AltmetricsCOVID["name"]), RDF.type, IOnt.Metric))
                        g.add((indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_" + name + "_value")))
                        g.add((URIRef(AltmetricsCOVID+doi+"_" + name + "_value"), RDF.type, IOnt.IndicatorValue))
                        g.add((URIRef(AltmetricsCOVID+doi+"_" + name + "_value"), IOnt.indicatorValue, Literal(total, datatype=XSD.integer)))
                        
                        if "count_types" in cat:
                            
                            for m in cat["count_types"]:
                                
                                name_2 = m["name"].lower()
                                total_2 = m["total"]
                                
                                level_2_indicator = URIRef(AltmetricsCOVID+doi+"_" + name_2)
                                g.add((level_2_indicator, RDF.type, IOnt.Indicator))
                                g.add((level_2_indicator, RDFS.label, Literal(m["name"])))
                                g.add((indicator, IOnt.hasSource, AltmetricsCOVID.plumx))
                                g.add((indicator, IOnt.hasSubIndicator, level_2_indicator))
                                g.add((indicator, IOnt.basedOnMetric, URIRef(AltmetricsCOVID["name"])))
                                g.add((level_2_indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_" + name_2 + "_value")))
                                g.add((URIRef(AltmetricsCOVID+doi+"_" + name_2 + "_value"), RDF.type, IOnt.IndicatorValue))
                                g.add((URIRef(AltmetricsCOVID+doi+"_" + name_2 + "_value"), IOnt.indicatorValue, Literal(total_2, datatype=XSD.integer)))
                                
                                level_3_indicator = URIRef(AltmetricsCOVID+doi+"_" + name_2 + "_source")
                                g.add((level_3_indicator, RDF.type, IOnt.Indicator))
                                g.add((level_3_indicator, RDFS.label, Literal(m["name"] + " source")))
                                g.add((indicator, IOnt.hasSource, AltmetricsCOVID.plumx))
                                g.add((level_2_indicator, IOnt.hasSubIndicator, level_3_indicator))
                                g.add((indicator, IOnt.basedOnMetric, URIRef(AltmetricsCOVID["name"])))
                                g.add((level_3_indicator, IOnt.hasIndicatorValue, URIRef(AltmetricsCOVID+doi+"_" + name_2 + "_value")))
        except:
            log.error("No altmetrics available for paper %s."%doi)
        
        return g


class DoiResolver():
    
    @staticmethod
    def resolve(papers_file : str, out_dois_file : str = None) -> DataFrame:
        
        
        cross_ref_api_client = CrossRefAPIClient()
        
        df = pd.read_csv(papers_file)
        
        data = []
        for row in df.values:
            paper_info = cross_ref_api_client.retrieve_doi(row[0], row[1], row[2])
            data.append(paper_info)
            
        dois = DataFrame(data=data)
        
        if out_dois_file is not None:
            dois_path = Path(out_dois_file)
            dois_parent_path = dois_path.parent
        
            if not os.path.exists(dois_parent_path):
                os.makedirs(dois_parent_path)
        
            dois.to_csv(out_dois_file)
            
        return dois
    
    
class MetadataBuilder():
    
    @staticmethod
    def build(dois_file: str, out_metadata_file : str = None) -> DataFrame:
        
        
        cross_ref_api_client = CrossRefAPIClient()
        
        df = pd.read_csv(dois_file)
        
        data = []
        for row in df.values:
            print(row[1])
            paper_info = cross_ref_api_client.retrieve_metadata(row[1])
            data.append(paper_info)
            
        metadata = DataFrame(data=data)
        
        if out_metadata_file is not None:
            metadata_path = Path(out_metadata_file)
            metadata_parent_path = metadata_path.parent
        
            if not os.path.exists(metadata_parent_path):
                os.makedirs(metadata_parent_path)
        
            metadata.to_csv(out_metadata_file)
            
        return metadata

class KGBuilder():
    
    @staticmethod
    def build(dest_folder:str, dois: DataFrame):
        
        scopus_dest = os.path.join(dest_folder, "scopus")
        plumx_dest = os.path.join(dest_folder, "plumx")
        if not os.path.exists(dest_folder):
            os.makedirs(scopus_dest)
            os.makedirs(plumx_dest)
        
        scopus_api_client = ScopusAPIClient(conf.INSTTOKEN, conf.APIKEY)
        for row in dois.values:
            doi = row[0]
            data_time = row[1]
            timestamp = row[2]
            
            paper = Paper(doi, data_time, timestamp)
            log.info("Processing paper with DOI %s", doi)
            doi_file = doi.replace("/", "_")
            
            g = scopus_api_client.citation_count(paper)
            
            out = os.path.join(scopus_dest, doi_file + ".nt")
            with codecs.open(out, 'w', encoding='utf8') as out_file:
                out_file.write(g.serialize(format="ntriples").decode('utf-8'))
                
            g = scopus_api_client.plum_x(paper)
            
            out = os.path.join(plumx_dest, doi_file + ".nt")
            with codecs.open(out, 'w', encoding='utf8') as out_file:
                out_file.write(g.serialize(format="ntriples").decode('utf-8'))
            

class Main:

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-o", "--output", dest="output",
                    help="Output folder. If no choice is provided then standard output is assumed as default.", metavar="FOLDER")
        parser.add_argument("-i", "--paper-info", dest="papers",
                    help="List of paper info. Information about a paper contains the authors, year, and title.")
        parser.add_argument("-d", "--dois", dest="dois",
                    help="The inpout CSV file containing the list of DOIS.")

        self.__args = parser.parse_args()
        
        self.__parser = parser
        
        log.basicConfig(level=conf.LOG_LEVEL) 
        
    def execute(self):
        
        if self.__args.dois is None and self.__args.papers is None:
            log.error(self.__parser.print_help())
            sys.exit(1)
        elif self.__args.dois is not None:
            dois = pd.read_csv(self.__args.dois)
        else:
            log.info("Retrieving DOIs associated with the papers provided as input.")
            dois = DoiResolver.resolve(self.__args.papers, r'./dois.csv')
            
        log.info("Building the Knowledge Graph.")
        KGBuilder.build(self.__args.output, dois)

if __name__ == '__main__':
    main = Main()
    main.execute()
    
    #cross_ref_api_client = CrossRefAPIClient()
    #cross_ref_api_client.retrieve_metadata("10.1001/jama.2020.3151")
    
    #MetadataBuilder.build(r'../dataset/dois_davide_golinelli.csv', "metadata.csv")
    
    #DoiResolver.resolve(r'./papers.csv', r'./dois.csv')
    
    #cross_ref_api_client = CrossRefAPIClient()
    #doi = cross_ref_api_client.retrieve_doi('C. Marasca', '2020', 'Telemedicine and support groups in order to improve the adherence to treatment and health related quality of life in patients affected by inflammatory skin conditions during COVID‐19 emergency')
    #print("DOI", doi)

    '''  
    scopus_api_client = ScopusAPIClient("f461367d15f83a8273b04e54ed983158", "5953888c807d52ee017df48501d3e598")


    #print(g.serialize(format='turtle', encoding='utf-8').decode('utf-8'))

    g = scopus_api_client.plum_x("10.1001/jama.2020.3151")

    print(g.serialize(format='turtle', encoding='utf-8').decode('utf-8'))

    #KGBuilder.build('out', r'../dataset/dois_davide_golinelli2.csv')
    '''


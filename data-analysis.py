'''
Created on May 8, 2020

The DataManager requires an active SPARQL endpoint that exposes the knowledge graph built with indicators-gathering.py.

@author: Andrea Giovanni Nuzzolese
'''
from argparse import ArgumentParser
from datetime import datetime
import math
import os
import traceback

from SPARQLWrapper.Wrapper import SPARQLWrapper, JSON
from numpy import source
from pandas.core.frame import DataFrame
from scipy import stats

import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

import conf

class DataManager():
    
    __SPARQL_INDICATORS = '''
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX scho: <https://w3id.org/scholarlydata/ontology/conference-ontology.owl#>
                PREFIX schoi: <https://w3id.org/scholarlydata/ontology/indicators-ontology.owl#>
                PREFIX dc: <http://purl.org/dc/elements/1.1/>
                SELECT ?DOI ?Metric ?Metric_Type ?Source (STR(?value) AS ?Score)
                WHERE{
                    ?paper a scho:Document;
                        schoi:hasIndicator ?indicator1 ;
                        dc:created ?timestamp .
                    BIND(REPLACE(STR(?paper), "https://doi.org/", "") AS ?DOI)
                    ?indicator1 rdfs:label ?Metric;
                        schoi:hasSubIndicator  ?indicator2 .   
                    ?indicator2 rdfs:label ?Metric_Type;       
                        schoi:hasSubIndicator ?indicator3 .   
                    ?indicator3 rdfs:label ?Source;       
                    schoi:hasIndicatorValue/schoi:indicatorValue ?value . 
                }
                '''
    
    __SPARQL_PUBLICATION_DATES = '''
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX scho: <https://w3id.org/scholarlydata/ontology/conference-ontology.owl#>
                PREFIX schoi: <https://w3id.org/scholarlydata/ontology/indicators-ontology.owl#>
                PREFIX dc: <http://purl.org/dc/elements/1.1/>
                SELECT ?DOI (STR(?timestamp) AS ?PubDate)
                WHERE{
                    ?paper a scho:Document;
                        schoi:hasIndicator ?indicator1 ;
                        dc:created ?timestamp .
                    BIND(REPLACE(STR(?paper), "https://doi.org/", "") AS ?DOI) 
                }
                '''
    
    def __init__(self, triplestore: str, out_folder: str):
        self.__out_folder = out_folder
        self.__triplestore = triplestore
        self.__df = self.__query_kb()
        self.__pub_dates = self.__query_publication_dates().set_index("DOI").astype("int32")
        dois = self.__df["DOI"]
        self.__dois = DataFrame(data=dois * len(dois))
        self.__metric_2_metric_type = self.__df[["Metric", "Metric Type"]].drop_duplicates().sort_values(by=["Metric", "Metric Type"])
        self.__metric_type_2_source = self.__df[["Metric Type", "Source"]].drop_duplicates().sort_values(by=["Metric Type", "Source"])
        
        categories = self.__metric_2_metric_type["Metric"].unique()
        metrics = self.__metric_2_metric_type["Metric Type"].unique()
        sources = self.__metric_type_2_source["Source"].unique()
        
        self.__df_categories = self.__reorganise_data("Metric", categories).fillna(0)
        self.__df_metrics = self.__reorganise_data("Metric Type", metrics).fillna(0)
        self.__df_sources = self.__reorganise_data("Source", sources).fillna(0)
    
    def __query_kb(self) -> DataFrame:
        
        rows_list = []
        sparql_wrapper = SPARQLWrapper(self.__triplestore)
        sparql_wrapper.setQuery(self.__SPARQL_INDICATORS)
        sparql_wrapper.setReturnFormat(JSON)
        try :
            results = sparql_wrapper.query().convert()
            
            for result in results["results"]["bindings"]:
                dict = {
                        "DOI": result["DOI"]["value"],
                        "Metric": result["Metric"]["value"],
                        "Metric Type": result["Metric_Type"]["value"],
                        "Source": result["Source"]["value"],
                        "Score": result["Score"]["value"]
                        }
                
                rows_list.append(dict)
        
        except :
            log.error("An error occurred while querying the SPARQL endpoint.")
            traceback.print_exc()
            
        return pd.DataFrame(rows_list)
    
    
    def __query_publication_dates(self) -> DataFrame:
        
        rows_list = []
        sparql_wrapper = SPARQLWrapper(self.__triplestore)
        sparql_wrapper.setQuery(self.__SPARQL_PUBLICATION_DATES)
        sparql_wrapper.setReturnFormat(JSON)
        try :
            results = sparql_wrapper.query().convert()
            
            for result in results["results"]["bindings"]:
                dict = {
                        "DOI": result["DOI"]["value"],
                        "PubDate": result["PubDate"]["value"][:-3]
                        }
                
                rows_list.append(dict)
        
        except :
            log.error("An error occurred while querying the SPARQL endpoint.")
            traceback.print_exc()
            
        return pd.DataFrame(rows_list)
    
    
    def __reorganise_data(self, metric_view, metrics):
        df = self.__df.groupby(by=["DOI", metric_view]).sum()
        #print(df.loc["10.1001/jama.2020.0757"].index)
        
        df_1 = pd.DataFrame()
        
        for doi in df.index.levels[0]:
            ff = {"DOI": doi}
        
            for cat in metrics:
                
                if cat in df.loc[doi].index:
                    a = df.loc[doi, cat]["Score"]
                    ff.update({cat: a})
                    
                else: 
                    ff.update({cat: 0})
        
            
            df_1 = df_1.append(ff, ignore_index=True)
            #df_1 = pd.DataFrame.from_dict(ff)
        df_1.set_index("DOI", inplace=True)
        
        df =  df_1.join(self.__dois.set_index("DOI"), on="DOI", how="outer", lsuffix="_l", rsuffix="_r")
        
        return df.astype('float64') 
    
    def get_df_categories(self):
        return self.__df_categories
    
    def get_df_metrics(self):
        return self.__df_metrics
    
    def get_df_sources(self):
        return self.__df_sources
    
    def get_metrics_by_category(self, category):
        m = self.__metric_2_metric_type[self.__metric_2_metric_type["Metric"] == category]["Metric Type"]
        return self.__df_metrics[m.values]
    
    def get_sources_by_metric(self, metric):
        m = self.__metric_type_2_source[self.__metric_type_2_source["Metric Type"] == metric]["Source"]
        return self.__df_sources[m.values]
    
    def categories(self):
        return self.__metric_2_metric_type["Metric"].unique()
    
    def metrics(self):
        return self.__metric_2_metric_type["Metric Type"].unique()
    
    def sources(self):
        return self.__metric_type_2_source["Source"].unique()
    
    def get_dois(self):
        return self.__dois
    
    def get_publication_dates(self):
        return self.__pub_dates
    
    def get_df(self):
        return self.__df
    
    def get_sparql_endpoint(self):
        return self.__triplestore
    
    def get_out_folder(self):
        return self.__out_folder
    
    
class DataAnalyser():
    
    def __init__(self, data_manager: DataManager):
        self.__data_manager = data_manager
        
    def to_timestamp(self, r):
        sss = datetime.fromtimestamp(r).strftime('%Y-%m-%d')
        return sss
        
    def cis(self, indicators=None):
        if indicators is None:
            df = self.__data_manager.get_df_categories()[self.__data_manager.get_df_categories().columns]
        else:
            df = self.__data_manager.get_df_categories()[indicators]
            
        zs = pd.DataFrame(stats.zscore(df), index=df.index, columns=[df.columns])
        df = zs.mean(axis=1)
        
        #df.drop_duplicates(inplace=True)
        df_1 = pd.DataFrame(df.index, columns=["DOI"])
        df_1["Score"] = df.values
        df_1.set_index("DOI", inplace=True)
        
        
        df_1= df_1.join(self.__data_manager.get_publication_dates(), how="inner")
        df_1["PubDate"] = df_1["PubDate"].apply(lambda x: self.to_timestamp(x)).values
        
        df_1.sort_values(by="PubDate", inplace=True)
        
        dates = df_1["PubDate"].unique()
        
        
        d = 0
        n_dates = []
        for date in dates:
            date_parts = date.split("-")
            d_tmp = int(date_parts[2])
            if d != 0:
                while d_tmp > (d+1):
                    d_1 = d+1
                    n_date = '%s-%s-%s'%(date_parts[0], date_parts[1], str(d_1)) 
                    n_dates.append(n_date)
                    d = d_1
                d += 1
            else:
                d = d_tmp
                
            n_dates.append(date)
            
        dates_df = pd.DataFrame(np.array(n_dates), columns=["PubDate"])
        
        df_1 = dates_df.join(df_1.reset_index().set_index("PubDate"), how="outer", on="PubDate", lsuffix="_l", rsuffix="_r")
        #df_1.fillna(nan, inplace=True)
        
        df_1.sort_values(by="PubDate", inplace=True)
        
        df_1.set_index("DOI", inplace=True)
        
        
        sn.set_style("whitegrid")
        
        dates = df_1["PubDate"]
        
        date_dict = dict()
        reverse_date_dict = dict()
        index = 0
        for date in dates.unique():
            date_dict.update({date: str(index)})
            reverse_date_dict.update({str(index): date})
            index += 1
            
        
        #chart = df_1.plot.scatter(x="Timestamp", y="Score")
        #df_1 = df_1.astype({"Timestamp": "str"})
        
        
        
        df_1["PubDate"] = df_1.apply(lambda x: date_dict.get(x["PubDate"]), axis=1)
        
        df_1 = df_1.astype({"PubDate": "int32"})
        chart = sn.scatterplot(x="PubDate", y="Score", data=df_1)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        
        quant_95 = df_1["Score"].quantile(0.95)
        chart.axhline(y=float(quant_95), xmin=0, xmax=1, linewidth=3)
        
        df_1 = df_1.dropna()
        
        out_folder = os.path.join(self.__data_manager.get_out_folder(), "cis")
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        out_csv = os.path.join(out_folder, "cis.csv")
        out_png = os.path.join(out_folder, "cis.png")
        
        plt.xticks(list(map(int, list(reverse_date_dict.keys()))), list(date_dict.keys()))
        
        
        
        if indicators is not None:
            self.label_cis_restricted_point(df_1, plt.gca(), quant_95)
            #df_1["PubDate"] = df_1["Timestamp"].apply(lambda x : self.to_timestamp(int(x)))
            df_1.to_csv(out_csv)
            plt.ylabel("$CIS_{I'}$")
        else:
            self.label_cis_all_point(df_1, plt.gca(), quant_95)
            #df_1["PubDate"] = df_1["Timestamp"].apply(lambda x : self.to_timestamp(int(x)))
            df_1.to_csv(out_csv)
            plt.ylabel("$CIS_{I}$")
        
        
        plt.gcf().set_size_inches(10.5, 5.5)
        plt.tight_layout()
        plt.xlabel("Publication date")
        
        fig = plt.savefig(out_png, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.3,
                    frameon=None)
        plt.close()
        
    
    def label_cis_all_point(self, doi, ax, t):
        
        #ax.text(4, t-0.1, r'95% quantile, i.e. $CIS_{I} \geq ' + str("{:.2f}".format(t) + '$'),
        ax.text(2, t+0.22, r'95% quantile, i.e. $CIS_{I} \geq ' + str("{:.2f}".format(t) + '$'),
                horizontalalignment='center',
                verticalalignment='top',
                multialignment='center')
        
        for i, point in doi.iterrows():
            if point["Score"] > t:
                if str(point.name).endswith("2391"):
                    ax.text(point['PubDate']-15.0, point['Score'], "  [%.2f] %s"%(point['Score'], str(point.name)))
                else:
                    ax.text(point['PubDate'], point['Score'], "  [%.2f] %s"%(point['Score'], str(point.name)))
            elif math.isnan(point["Score"]): 
                ax.set_visible(True)
                
    def geometric_space_x_y(self):
        
        
                
                
    def __geometric_space_x_y(self, x, y, lp_fun=None):
        
        out_folder = os.path.join(self.__data_manager.get_out_folder(), "geometic_space")
        
        out_file = os.path.join(out_folder, "%s_%s.csv"%(x,y))
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        df = self.__data_manager.get_df_categories()
        zs = pd.DataFrame(stats.zscore(df), index=df.index, columns=[df.columns], dtype="float")
        #zs.plot.scatter(x=x, y=y, s=25, grid=True)
        
        #print(zs)
        
        x1, y1 = np.random.randint(-8,8,5), np.random.randint(-8,8,5)
        
        a = [zs[x].values, zs[y].values]
        
        #print(a)
        
        vmax = np.abs(np.concatenate([x1,y1])).max() + 5
        vmax_1 = np.abs(np.concatenate(a)).max() + 1
        
        #t = [[0.0, float(zs[y].max()/2)][float(zs[x].max()), float(zs[y].max()/2)]]
        
        
        extent = [vmax*-1,vmax, vmax*-1,vmax]
        extent_1 = [vmax_1*-1,vmax_1, vmax_1*-1,vmax_1]
        
        #print(extent_1)
        
        fig, ax = plt.subplots(1,1)
        
        quant_x = zs[x].quantile(0.95, interpolation="midpoint")
        quant_y = zs[y].quantile(0.95, interpolation="midpoint")
        
        
        ax.scatter(zs[x],zs[y], marker='s', s=30, c='r', edgecolors='red', lw=1)
        
        ax.autoscale(False)
        ax.scatter(zs[x],zs[y], marker='s', s=30, c='r', edgecolors='red', lw=1)
        ax.axhline(y=float(quant_y), xmin=0, xmax=float(zs[x].max()), linewidth=3)
        ax.axvline(x=float(quant_x), ymin=0, ymax=float(zs[y].max()), linewidth=3)
        
        
        #arr = np.array([[0,3,1,3,2,3],[1,3,2,3,3,3],[0,3,1,3,2,3],[1,3,2,3,3,3]])
        #ax.imshow(arr, extent=extent_1, cmap=plt.cm.Greys, interpolation='none', alpha=.1)
        
        if lp_fun is not None:
            lp_fun(zs, quant_x, quant_y, plt.gca())
        
        ax.grid(True)
        plt.gcf().set_size_inches(10.5, 5.5)
        
        plt.tight_layout()
        fig = plt.savefig(out_file, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.3,
                    frameon=None)
                

class Main:

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("output", help="The folder used for storing the output of data analysis.")
        parser.add_argument("sparql_endpoint", help="The SPARQL endpoint that exposes that knowledge graph for queries.")

        self.__args = parser.parse_args()
        
        if not os.path.exists(self.__args.output):
            os.makedirs(self.__args.output)
        
        self.__parser = parser
        
        self.__data_manager = DataManager(self.__args.sparql_endpoint, self.__args.output)
        
        log.basicConfig(level=conf.LOG_LEVEL)
        
    def execute(self):
        data_analyser = DataAnalyser(self.__data_manager)
        data_analyser.cis()
        
        
if __name__ == '__main__':
    Main().execute()
        

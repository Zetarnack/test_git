# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:25:00 2024

@author: tp3682
"""

import urllib
import streamlit as st
import os
# import tdqm #Ajouter des barres de chargement
import requests
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from math import *
import numpy as np
from io import BytesIO


warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning) Ajouter cette exception


import pandas as pd
# from .error_hubeau import *


def get_stations_hydro_from_hubeau_via_code_dept(code_departement, size=10000):
    url_api = r"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?code_departement={code_dept}&format=json&size={size}".format(
        code_dept=urllib.parse.quote_plus(str(code_departement)), size=size)
    api_station = requests.get(url_api)
    json_data = api_station.json()
    df = pd.DataFrame(json_data["data"])
    while json_data["count"] != len(df):
        res_api = requests.get(json_data["next"])
        json_data = res_api.json()
        df2 = pd.DataFrame(json_data["data"])
        df = df.append(df2, ignore_index=True)
        df = df.drop_duplicates(keep="first", inplace=False)
    df = df.drop(columns=['geometry'], axis=1)
    return df


def get_chroniques_hydro_into_df(code_site, size=20000):
    url_api = r"https://hubeau.eaufrance.fr/api/v1/hydrometrie/obs_elab?code_entite={code_entite}&format=json&size={size}".format(
        code_entite=urllib.parse.quote_plus(code_site), size=size)
    api_station = requests.get(url_api)
    json_data = api_station.json()
    df = pd.DataFrame(json_data["data"])
    df=df.rename(columns={"resultat_obs_elab":"Qm"})
    df['Qm']=df['Qm']/1000
    # return df
    if json_data["count"] <= 20000:
        return df
    else:
        while json_data["count"] != len(df):
            res_api = requests.get(json_data["next"])
            # print(res_api, len(df), sep='\n')
            json_data = res_api.json()
            df2 = pd.DataFrame(json_data["data"])
            df = pd.concat([df, df2], ignore_index=True)
            df = df.drop_duplicates(keep="first", inplace=False)
        return df

def get_flood_discharge_data(station,date):
    debut=str(date-timedelta(days=20))
    fin=str(date+timedelta(days=20))
    
    
    url=r"https://www.hydro.eaufrance.fr/sitehydro/ajax/"+station+"/series?hydro_series%5BstartAt%5D="+debut[8:
        10]+"%2F"+debut[5:7]+"%2F"+debut[:4]+"&hydro_series%5BendAt%5D="+fin[8:
        10]+"%2F"+fin[5:7]+"%2F"+fin[:4]+"&hydro_series%5BvariableType%5D=simple_and_interpolated_and_hourly_variable&hydro_series%5BsimpleAndInterpolatedAndHourlyVariable%5D=Q&hydro_series%5BstatusData%5D=most_valid"
    response=requests.get(url,verify=False)
    data=response.json()
    
    title=data['series']['title'][-41:-5]
    df=pd.DataFrame(data['series']['data'])
    df['v']=df['v']/1000
    df=df.rename(columns={"v":"Q (m³/s)","t":"Date"})
    # df["Date"]=pd.to_datetime(df["Date"])
    cols=df.columns.tolist()
    cols[0], cols[1] = cols[1], cols[0]
    df=df[cols]
    
    
    return(title,df)

def save_plot(df,title) :
    df['Date']=df["Date"]=pd.to_datetime(df["Date"])
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Q (m³/s)'], marker='o', linestyle='-')
    plt.title('Évolution temporelle des débits - Du '+ title)
    plt.xlabel('Date')
    plt.ylabel('Débit (m³/s)')
    plt.grid(True)
    # Format de la date sur l'axe x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))

    # Rotation des labels de l'axe x pour une meilleure lisibilité
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    
    # Sauvegarde du graphique dans un objet BytesIO
    imgdata = BytesIO()
    plt.savefig(imgdata, format='png')
    imgdata.seek(0)
    plt.close()
    
    return(imgdata)

def get_baseflow (station):
    """Qm est le résultat de la fonction get_chroniques_hydro_into_df
    Pour une station donnée, renvoit le débit de base en calculant le débit moyen sur la période
    des 60 jours consécutifs les plus secs mesurés par la station
    """
    #Détermination du débit de base
    Qm=get_chroniques_hydro_into_df(station)
    Qm=Qm.loc[Qm['grandeur_hydro_elab']=='QmJ'] #Ne conserver que les débits moyens journaliers
    Qm.loc[np.isnan(Qm['Qm']),'Qm']=Qm['resultat_obs_elab']/1000 #Compléter les valeurs récentes
    # Calculer la somme cumulée sur 60 jours glissants, en excluant les périodes avec NaN
    Qm['Cumul_60j'] = Qm['Qm'].rolling(window=60, min_periods=60).sum()
    # Supprimer les périodes où il y a des NaN dans la fenêtre de 60 jours
    Qm['NaN_Count_60j'] = Qm['Qm'].rolling(window=60).apply(lambda x: np.isnan(x).sum(), raw=True)
    Qm = Qm[Qm['NaN_Count_60j'] == 0]  # Ne garder que les périodes sans NaN
    # Trouver l'indice de la période la plus sèche (cumul le plus bas)
    idx_min = Qm['Cumul_60j'].idxmin()
    # Identifier la période correspondante
    Qmin_list=[]
    for k in range (idx_min, idx_min+59):
        Qmin_list.append(Qm['Qm'][k])
    Qbaseflow=np.mean(Qmin_list)
    
    return(Qbaseflow)

# Fonction exponentielle pour la récession
def recession_curve(t, Q0, alpha):
    return Q0 * np.exp(-alpha * t)

# Fonction pour vérifier la décroissance des débits après une valeur donnée
def check_decreasing_after_index(df, idx):
    return all(df['Q (m³/s)'].loc[idx+1:] < df['Q (m³/s)'].loc[idx])

def params_hydrograph (df,Qbaseflow,S):
        
    #Analyse hydrogramme de crue
    Qp=max(df['Q (m³/s)'])
    Date_Qp = df['Date'][df['Q (m³/s)'].idxmax()]
    Dmax=1.5*(5+log(S/(1.609)**2))
    # 1. Déterminer le début de la crue
        # 1.1. Filtrer les débits avant le pic de crue
    df_avant_pic = df[df['Date'] < Date_Qp]
        # 1.2. Trouver les minima locaux avant le pic de crue
    df_avant_pic['MinLocal'] = (df_avant_pic['Q (m³/s)'] < df_avant_pic['Q (m³/s)'].shift(1)) & (df_avant_pic['Q (m³/s)'] < df_avant_pic['Q (m³/s)'].shift(-1))
        # 1.3. Filtrer selon les conditions : débit < moitié du pic et Dmax
    df_candidates = df_avant_pic[(df_avant_pic['MinLocal']) & (df_avant_pic['Q (m³/s)'] < Qp / 2)]
    df_candidates['Duree'] = (Date_Qp - df_candidates['Date']).dt.days
    df_final = df_candidates[df_candidates['Duree'] <= Dmax]
        # 1.4. Trouver le minimum local le plus proche du pic de crue
    if not df_final.empty:
        Debut_crue = df_final.iloc[-1]
    else :
        return (np.nan, np.nan,np.nan,np.nan) #Quitter la fonction si le début ou la fin de la crue n'est pas défini
    
    # 2. Déterminer la fin de la crue
        # 2.1. Filtrer les débits avant le pic de crue
    df_apres_pic = df[df['Date'] > Date_Qp]
        # 2.2. Trouver les minima locaux après le pic de crue
    df_apres_pic['MinLocal'] = (df_apres_pic['Q (m³/s)'] < df_apres_pic['Q (m³/s)'].shift(1)) & (df_apres_pic['Q (m³/s)'] < df_apres_pic['Q (m³/s)'].shift(-1))
    
        # 2.3. Filtrer selon les conditions : débit < moitié du pic et Dmax
    df_candidates = df_apres_pic[(df_apres_pic['MinLocal']) & (df_apres_pic['Q (m³/s)'] < Qp / 2)]
    df_candidates['Duree'] = (df_candidates['Date']-Date_Qp).dt.days
    df_final = df_candidates[df_candidates['Duree'] <= Dmax]
        # 1.4. Trouver le minimum local le plus proche du pic de crue
    if not df_final.empty:
        Fin_crue = df_final.iloc[0]
    else :
        return (np.nan, np.nan,np.nan,np.nan)
    
    t_montee=(Date_Qp-Debut_crue['Date'])/timedelta(days=1)
    t_base=(Fin_crue['Date']-Debut_crue['Date'])/timedelta(days=1)
    
    # 3. paramètres de l'hydrogramme unitaire
    facteur_forme=t_base/t_montee
    df_unitaire = df['Date'][np.logical_and(df['Date'] >= Debut_crue['Date'], df['Date'] <= Fin_crue['Date'])].reset_index(drop=True)
    Date_unitaire=[Debut_crue['Date']]
    Q_unitaire=[Debut_crue['Q (m³/s)']]
    t_unitaire=[0]
    for k in range (1,len(df_unitaire)):
        t=(df_unitaire[k]-Debut_crue['Date'])/timedelta(days=1)
        t_unitaire.append(t)
        Date_unitaire.append(df_unitaire[k])
        Q_unitaire.append(Debut_crue['Q (m³/s)']+(Qp-Debut_crue['Q (m³/s)'])*pow((t/t_montee)*exp(1-t/t_montee),facteur_forme))
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Q (m³/s)'], linestyle='-', label='Données observées')
        #3.0 Plot hydrogramme unitaire
    plt.plot(Date_unitaire, Q_unitaire, linestyle='--', label='Hydrogramme unitaire')
    
    
    # Tracé de la ligne verticale en pointillée
    plt.axvline(x=Debut_crue['Date'], color='gray', linestyle='--')
    plt.axvline(x=Fin_crue['Date'], color='gray', linestyle='--')
    plt.axhline(y=Qbaseflow, color='green', linestyle='--', label='Débit de base')
    
    plt.xlabel('Date')
    plt.ylabel('Débit (m³/s)')
    plt.title('Paramètres')
    plt.legend()
    plt.grid(True)
    # Sauvegarde du graphique dans un objet BytesIO
    imgdata = BytesIO()
    plt.savefig(imgdata, format='png')
    imgdata.seek(0)
    plt.plot()
    plt.close()
    
    #4 Calcul des paramètres
    
        #4.1 Estimation du volume
    # t_crue=[]
    # for k in range (len(df['Date'][df.index[df['Date'] == Debut_crue['Date']][0]:df.index[df['Date'] == Fin_crue['Date']][0]])):
    #     t_crue.append((df['Date'][df.index[df['Date'] == Debut_crue['Date']][0]+k]-df['Date'][df.index[df['Date'] == Debut_crue['Date']][0]])/timedelta(days=1))
    # Donnees_crue=df['Q (m³/s)'][df.index[df['Date'] == Debut_crue['Date']][0]:df.index[df['Date'] == Fin_crue['Date']][0]]
    # Diff_crue_base=Donnees_crue.apply(lambda x: x - Qbaseflow if x > Qbaseflow else 0)
    # t_crue=np.array(t_crue)*3600*24
    # if Qp>Qbaseflow :
    #     Integrale_diff_crue_base=np.trapz(Diff_crue_base,t_crue)
    # else : Integrale_diff_crue_base=0
    # Integrale_Diff_tarissement_base=np.trapz(Q_tarissement-Qbaseflow,np.array(t_tarissement)*3600*24)
    # Integrale_Diff_debut_base=np.trapz(Q_debut-Qbaseflow,np.array(t_debut)*3600*24)
    # Volume=Integrale_diff_crue_base-Integrale_Diff_debut_base+Integrale_Diff_tarissement_base
    Volume=np.trapz(Q_unitaire-Qbaseflow,np.array(t_unitaire)*3600*24)
    
    Param_crue = {}
    Param_crue['Qbaseflow (m³/s)']=Qbaseflow
    Param_crue['Qp (m³/s)']=Qp
    Param_crue['Date Qp']= str(Date_Qp)
    Param_crue['Date début']=str(Debut_crue['Date'])
    Param_crue['Fin début']=str(Fin_crue['Date'])
    Param_crue['Temps de montée (jours)']=t_montee
    Param_crue['Temps de base (jours)']=t_base
    Param_crue['Volume (m³)']=Volume

    return(Param_crue,imgdata,t_unitaire,Q_unitaire)

#Lecture du csv Gumbel d'Hydroportail
#A rentrer par l'utilisateur
# print('Rentrer le chemin vers le csv des valeurs utilisées pour effectuer l_ajustement de Gumbel de Hydroportail')
# csv_file=input()
# print('Rentrer la superficie du bassin versant associé à la station (km²)')
# S=input()

csv_file=r'C:/Users/tp3682/Desktop/Hub_eau/Q-X-(CRUCAL)_Gumbel_J4514010_11-07-1973_25-08-2024_1_non-glissant_X_pre-valide-et-valide_Echantillon.csv'
S=21 #Surface du BV en km²

df_general=pd.read_csv(csv_file)
station = (csv_file.split('/')[-1]).replace('Q-X-(CRUCAL)_Gumbel_', '').split('_')[0]

#Extraction du débit de base avec l'API des débits moyens journaliers
# Qm=get_chroniques_hydro_into_df(station, size=20000) #débits moyens journalier
Qbaseflow=get_baseflow (station) #Débit de base

#Extraction des crhoniques avec l'url
title_list=[]
df_list=[]
T_list=[]
graph_list=[]
graph2_list=[]
Params_list=[]
for k in range(len(df_general)):
    if df_general['Qualification(s)'][k] == 20 : #Valeur bonne
        date=datetime.strptime(df_general['Date(s)'][k],"%Y-%m-%dT%H:%M:%S.%fZ")
        title,df=get_flood_discharge_data(station,date)
        T_list.append(1/(1-df_general['Fréquence au non dépassement'][k]))
        df['Q (m³/s)'] = pd.to_numeric(df['Q (m³/s)'], errors='coerce')
        df['Date']=pd.to_datetime(df['Date'])
        Params,graph2,t_unitaire,Q_unitaire=params_hydrograph (df,Qbaseflow,S)
        Params_list.append(Params)
        title_list.append(title)
        df_list.append(df)
        graph_list.append(save_plot(df,title))
        graph2_list.append(graph2)


#Créaction d'un Excel de synthèse
        
with pd.ExcelWriter('C:/Users/tp3682/Desktop/Hub_eau/Crues_'+station+'.xlsx',  engine='xlsxwriter') as writer:
    j=0
    for sheet_name, df in zip(title_list, df_list):
        sheet_name=sheet_name.replace("/","_").split(" ")[0]+' '+sheet_name.replace("/","_").split(" ")[3]
        df['Date'] = df['Date'].dt.tz_localize(None)
        df.to_excel(writer, sheet_name, index=False)
        workbook=writer.book
        worksheet=writer.sheets[sheet_name]
        worksheet.insert_image('K3', 'Évolution temporelle des débits', {'image_data':graph_list[j]})
        worksheet.write('L33','Période de retour (années)')
        worksheet.write('M33',T_list[j])
        if type((Params_list[j]))==dict :
            start_row = 35
            start_col = 11  # Colonne K correspond à 11ème colonne

            # Écrire les clés du dictionnaire dans la première colonne
            for i, (key, value) in enumerate(Params_list[j].items(), start=start_row):
                worksheet.write(i, start_col, key)  # Écrire la clé dans la colonne de départ
                worksheet.write(i, start_col + 1, value)  # Écrire la valeur dans la colonne suivante

            worksheet.insert_image('Q35', 'Évolution temporelle des débits', {'image_data':graph2_list[j]})
            worksheet.write('L46','Hydrogramme unitaire')
            worksheet.write('L47','t (jours)')
            worksheet.write('M47','Q (m³/s)')
            
            start_row = 48
            start_col = 11
            # Écrire les clés du dictionnaire dans la première colonne
            for i in range (len(Q_unitaire)):
                worksheet.write(start_row+i, start_col, t_unitaire[i])
                worksheet.write(start_row+i, start_col+1, Q_unitaire[i])
            
        j+=1
        
#interface streamlit
# Créez un menu déroulant pour sélectionner le DataFrame
df_dict=dict(zip(title_list,df_list))
selected_df_name = st.selectbox('Sélectionnez un DataFrame à afficher', options=list(df_dict.keys()))
# Récupérez le DataFrame sélectionné
selected_df = df_dict[selected_df_name]
# Affichez le DataFrame sélectionné
st.dataframe(selected_df)


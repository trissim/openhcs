import copy
import xlsxwriter
import string
import numpy as np
import pandas as pd
import sys

from openhcs.formats.experimental_layout_rows import ExperimentalLayoutRowRole




def read_results(results_path,scope=None):
    xls = pd.ExcelFile(results_path)
    if scope == "EDDU_CX5":
        raw_df = pd.read_excel(xls, 'Rawdata')
    elif scope == "EDDU_metaxpress":
        raw_df = pd.read_excel(xls, xls.sheet_names[0])
    else:
        print("microscope "+str(scope)+" not known. Exiting")
        sys.exit()
    return raw_df

def get_features(raw_df,scope=None):
    if scope == "EDDU_CX5":
        return get_features_EDDU_CX5(raw_df)
    if scope == "EDDU_metaxpress":
        return get_features_EDDU_metaxpress(raw_df)
    else:
        print("microscope "+str(scope)+" not known. Exiting")
        sys.exit()

def read_plate_layout(config_path):
    xls = pd.ExcelFile(config_path)
    df = pd.read_excel(xls, 'drug_curve_map',index_col=0,header=None)
    df = df.dropna(how='all')
    layout={}
    condition=None
    doses=None
    wells=None
    plate_groups=None
    N = None
    specific_N = None
    scope = None
    conditions=[]
    ctrl_wells=None
    ctrl_wells_aligned=None
    ctrl_groups=None
    ctrl_positions_replicates=None
    ctrl_positions=None

    def sanitize_compare(string1,string2):
        string1 = string1.lower()
        string2 = string2.lower()
        string1 = string1.replace('_','')
        string1 = string1.replace(' ','')
        string2 = string2.replace('_','')
        string2 = string2.replace(' ','')
        if string1[-1] != 's':
            string1 += 's'
        if string2[-1] != 's':
            string2 += 's'
        return string1 == string2

    for i,row in df.iterrows():
        #check max number of replicates
        row_role = ExperimentalLayoutRowRole(row.name)
        if row_role.is_replicate_count:
            N = int(row.iloc[0])
            for i in range(N):
                layout["N"+str(i+1)]={}
        #load microscope
        if sanitize_compare(row.name,'scope') or sanitize_compare(row.name,'microscope'):
            scope = row.iloc[0]

        #finished reading controls
        if sanitize_compare(row.name,'plate group') and ctrl_wells is not None:
            if ctrl_groups is None:
                ctrl_groups = []
            ctrl_groups += row.dropna().tolist()
            continue
#        if sanitize_compare(row.name,'plate group') and not ctrl_wells is None and not ctrl_groups is None:
#            ctrl_positions = []
#            for i in range(len(ctrl_wells_aligned)):
#                if not ctrl_well_replicates is None:
#                    ctrl_positions.append((ctrl_wells_aligned[i],ctrl_groups[i],ctrl_well_replicates[i]))
#                else:
#                    ctrl_positions = None
#            continue

        #get control wells
        if sanitize_compare(row.name,'control') or sanitize_compare(row.name,'control well'):
            if ctrl_wells is None:
                ctrl_wells = []
            ctrl_wells+=row.dropna().tolist()
            continue

        #get replicate for ctrl position
        if sanitize_compare(row.name,'group n'):
            if ctrl_positions_replicates is None:
                ctrl_positions_replicates = []
            if ctrl_wells_aligned is None:
                ctrl_wells_aligned = []
            ctrl_positions_replicates+=row.dropna().tolist()
            ctrl_wells_aligned += ctrl_wells
            continue

        #get new condition name
        #finished reading controls
        if sanitize_compare(row.name,'condition'):
            # make control well dict
            ctrl_positions = {"N"+str(i+1):[] for i in range(N)}
            for i in range(len(ctrl_wells_aligned)):
                if ctrl_positions_replicates is not None:
                    ctrl_positions["N"+str(ctrl_positions_replicates[i])].append((ctrl_wells_aligned[i],ctrl_groups[i]))
                    ctrl_wells = None
                else:
                    ctrl_positions = None

            #make dict[replicate][condition][dose]
            for i in range(N):
                if row.iloc[0] not in layout["N"+str(i+1)].keys():
                    layout["N"+str(i+1)][row.iloc[0]]={}
            condition=row.iloc[0]
            conditions.append(condition)
        if sanitize_compare(row.name,'dose'):
            doses=row.dropna().tolist()

        #if well is same for all Ns
        row_role = ExperimentalLayoutRowRole(row.name)
        if row_role.is_well_all_replicates:
            wells=row.dropna().tolist()
            specific_N = None
        # or not
        if row_role.is_well_specific_replicate:
            specific_N = int(row.name[-1])
            wells=row.dropna().tolist()

        # add plate group to wells from previous row
        if sanitize_compare(row.name,'plate group'):
            plate_groups=row.dropna().tolist()
            if specific_N == None:
                for i in range(N):
                    for y in range(len(doses)):
                        #add to all Ns
                        if doses[y] not in layout["N"+str(i+1)][condition].keys():
                            layout["N"+str(i+1)][condition][doses[y]]=[]
                        layout["N"+str(i+1)][condition][doses[y]].append((wells[y],plate_groups[y]))
            else:
                for y in range(len(doses)):
                    #add to specific N
                    if doses[y] not in layout["N"+str(specific_N)][condition].keys():
                        layout["N"+str(specific_N)][condition][doses[y]]=[]
                    layout["N"+str(specific_N)][condition][doses[y]].append((wells[y],plate_groups[y]))
    return scope, layout, conditions, ctrl_positions

def get_features_EDDU_CX5(raw_df):
    return raw_df.iloc[:,raw_df.columns.str.find("Replicate").argmax()+1:-1].columns

def get_features_EDDU_metaxpress(raw_df):
    feature_rows = raw_df[pd.isnull(raw_df.iloc[:,0])].iloc[0].tolist()[2:]
    return feature_rows

def create_well_dict(raw_df, wells=None,scope=None):
    if wells == None:
        rows=[string.ascii_uppercase[i] for i in range(8)]
        cols=[i+1 for i in range(12)]
        wells = []
        for row in rows:
            for col in cols:
                wells.append(str(row)+str(col).zfill(2))
    features = get_features(raw_df,scope=scope)
    return {well:{feature:None for feature in features} for well in wells}

if __name__ == "__main__":
    # This code only runs when the script is executed directly, not when imported as a module
    results_path="mx_results.xlsx"
    config_file="./config.xlsx"
    compiled_results_path="./compiled_results_normalized.xlsx"
    heatmap_path="./heatmaps.xlsx"

    scope, plate_layout, conditions, ctrl_positions=read_plate_layout(config_file)
    plate_groups=load_plate_groups(config_file)
    experiment_dict_locations=make_experiment_dict_locations(plate_groups,plate_layout,conditions)
    df = read_results(results_path,scope=scope)
    features = get_features(df,scope=scope)
    well_dict=create_well_dict(df,scope=scope)
    plates_dict=create_plates_dict(df,scope=scope)
    plates_dict = fill_plates_dict(df,plates_dict,scope=scope)
    experiment_dict_values=make_experiment_dict_values(plates_dict,experiment_dict_locations,features)
    if ctrl_positions is not None:
        experiment_dict_values=normalize_experiment(experiment_dict_values,ctrl_positions,features,plates_dict)
    feature_tables = create_all_feature_tables(experiment_dict_values,features)
    write_values_heat_map(plates_dict,features,heatmap_path)
    feature_tables_to_excel(feature_tables,compiled_results_path)

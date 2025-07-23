from skimage import io
import copy
import xlsxwriter
import string
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle
import pudb
import sys




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

def is_N_row(row_name):
    row_name = row_name.lower()
    is_N = False
    if row_name == "n" or row_name=="ns":
        is_N = True
    if row_name == "replicate" or row_name=="replicates":
        is_N = True
    return is_N


def is_well_all_replicates_row(row_name):
    row_name = row_name.lower()
    return row_name == "well" or row_name == "wells"

def is_well_specific_replicate_row(row_name):
    row_name = row_name.lower()
    if 'well' in row_name:
        return row_name[-1].isdigit()
    else: return False

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
        if not string1[-1] == 's': string1 +='s'
        if not string2[-1] == 's': string2 +='s'
        return string1 == string2

    for i,row in df.iterrows():
        #check max number of replicates
        if is_N_row(row.name):
            N = int(row.iloc[0])
            for i in range(N):
                layout["N"+str(i+1)]={}
        #load microscope
        if sanitize_compare(row.name,'scope') or sanitize_compare(row.name,'microscope'):
            scope = row.iloc[0]

        #finished reading controls
        if sanitize_compare(row.name,'plate group') and not ctrl_wells is None:
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
                if not ctrl_positions_replicates is None:
                    ctrl_positions["N"+str(ctrl_positions_replicates[i])].append((ctrl_wells_aligned[i],ctrl_groups[i]))
                    ctrl_wells = None
                else:
                    ctrl_positions = None

            #make dict[replicate][condition][dose]
            for i in range(N):
                if not row.iloc[0] in layout["N"+str(i+1)].keys():
                    layout["N"+str(i+1)][row.iloc[0]]={}
            condition=row.iloc[0]
            conditions.append(condition)
        if sanitize_compare(row.name,'dose'):
            doses=row.dropna().tolist()

        #if well is same for all Ns
        if is_well_all_replicates_row(row.name):
            wells=row.dropna().tolist()
            specific_N = None
        # or not
        if is_well_specific_replicate_row(row.name):
            specific_N = int(row.name[-1])
            wells=row.dropna().tolist()

        # add plate group to wells from previous row
        if sanitize_compare(row.name,'plate group'):
            plate_groups=row.dropna().tolist()
            if specific_N == None:
                for i in range(N):
                    for y in range(len(doses)):
                        #add to all Ns
                        if not doses[y] in layout["N"+str(i+1)][condition].keys():
                            layout["N"+str(i+1)][condition][doses[y]]=[]
                        layout["N"+str(i+1)][condition][doses[y]].append((wells[y],plate_groups[y]))
            else:
                for y in range(len(doses)):
                    #add to specific N
                    if not doses[y] in layout["N"+str(specific_N)][condition].keys():
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

def add_well_to_well_dict(wells,well_dict, raw_df):
    features = get_features(raw_df).columns
    for well in wells:
        well_dict[well]={feature:None for feature in features}
    return well_dict

def create_plates_dict(raw_df,scope=None):
    if scope == "EDDU_CX5":
        return create_plates_dict_EDDU_CX5(raw_df)
    if scope == "EDDU_metaxpress":
        return create_plates_dict_EDDU_metaxpress(raw_df)
    else:
        print("microscope "+str(scope)+" not known. Exiting")
        sys.exit()

def create_plates_dict_EDDU_metaxpress(raw_df):
    plate_names = raw_df[(raw_df == 'Plate Name').any(axis=1)].iloc[:,1].tolist()
    plate_dict = {plate_id:create_well_dict(raw_df,scope="EDDU_metaxpress") for plate_id in plate_names}
    return plate_dict

def create_plates_dict_EDDU_CX5(raw_df):
    plate_ids = raw_df['UniquePlateId'].tolist()
    plate_dict = {plate_id:create_well_dict(raw_df,scope="EDDU_CX5") for plate_id in plate_ids}
    return plate_dict

def indices_to_well(row,col,dim):
    rMax, cMax = dim[0],dim[1]
    col += 1
    total = row*cMax+col
    i=0
    i+=1
    offset = int((total-1)/(cMax)*i)
    rowIndex = str(chr(65 + offset))
    colIndex = str(total - (offset * (cMax)*i)).zfill(2)
    return rowIndex + str(colIndex)

def row_col_to_well(row,col):
    row_letter=chr(row+64)
    number=str(col).zfill(2)
    return row_letter+number

def well_to_num(well,dim):
    rMax, cMax = dim[0],dim[1]
    (rowIndex, colIndex) = (0,0)
    for i in range(0, len(well)):
        (left, right) = (well[:i], well[i:i+1])
        if right.isdigit():
            (rowIndex, colIndex) = (left, well[i:])
            break
    ascii_value = ord(rowIndex) - 65
    return ascii_value*(rMax+(4*i)) + int(colIndex)

def fill_plates_dict(raw_df,plates_dict,scope=None):
    features = get_features(raw_df,scope=scope)
    if scope == "EDDU_CX5":
        return fill_plates_dict_EDDU_CX5(raw_df,plates_dict,features)
    if scope == "EDDU_metaxpress":
        return fill_plates_dict_EDDU_metaxpress(raw_df,plates_dict,features)
    else:
        print("microscope "+str(scope)+" not known. Exiting")
        sys.exit()

def fill_plates_dict_EDDU_CX5(raw_df,plates_dict,features):
    for index,row in raw_df.iterrows():
        well = row_col_to_well(row[2],row[3])
        for feature in features:
            plates_dict[row[1]][well][feature]=row[feature]
    return plates_dict

def fill_plates_dict_EDDU_metaxpress(raw_df,plates_dict,features):
    df_col_names = raw_df.set_axis(["Well","Laser Focus"]+features, axis=1, inplace=False)
    plate_name=None
    start_collect=False
    for index,row in df_col_names.iterrows():
        if row[0] == "Barcode":
            start_collect=False
        if start_collect:
            for feature in features:
                plates_dict[plate_name][row[0]][feature]=row[feature]
        if row[0] == "Plate Name":
            plate_name=row[1]
        elif pd.isnull(row[0]):
            start_collect=True
    return plates_dict

def average_plates(plates,raw_df,scope=None):
    average_plate=create_well_dict(raw_df,scope=scope)
    features = get_features(raw_df)
    for well in average_plate.keys():
        for feature in features:
            average_value=0
            for plate in plates:
                average_value+=plate[well][feature]
            average_value=average_value/len(plates)
            average_plate[well][feature]=average_value
    return average_plate

def average_plates_all_replicates(plate_groups,plates_dict,raw_df):
    averaged_plates_dict = {replicate:None for replicate in plate_groups.keys()}
    for replicate in plate_groups.keys():
        one_replicate=average_plates_one_replicate(plate_groups[replicate],plates_dict,raw_df)
        averaged_plates_dict[replicate]=one_replicate
    return averaged_plates_dict

def average_plates_duplicate_rows(plate_groups,plates_dict,raw_df,wells_to_average=None,scope=None):
    features = get_features(raw_df,scope=scope)
    averaged_plates_dict={}
    for plate_name,plate in plates_dict.items():
        average_plate=create_well_dict(raw_df,scope=scope,wells=wells_to_average)
        for well in wells_to_average:
            average_plate=average_rows(plate,average_plate,well,features)
        averaged_plates_dict[plate_name]=average_plate
    return plates_dict

def average_rows(plate_dict,average_plate,well,features,num_rows_average=2):
    original_well=well
    wells_to_average = []
    wells_to_average.append(well)
    for i in range(num_rows_average-1):
        well_next_row = get_well_next_row(well)
        wells_to_average.append(well_next_row)
        well_next_row = well
    for feature in features:
        average_value=0
        for well in wells_to_average:
            average_value+=plate_dict[well][feature]
        average_value=average_value/num_rows_average
        average_plate[original_well][feature]=average_value
    return average_plate

def get_well_next_row(well):
    return chr(ord(well[0])+1)+well[1:]


def average_plates(plates,raw_df,scope=None):
    average_plate=create_well_dict(raw_df,scope=scope)
    features = get_features(raw_df)
    for well in average_plate.keys():
        for feature in features:
            average_value=0
            for plate in plates:
                average_value+=plate[well][feature]
            average_value=average_value/len(plates)
            average_plate[well][feature]=average_value
    return average_plate


def average_plates_one_replicate(averaged_plates_names_dict,plates_dict,raw_df):
    averaged_plates_dict = {plate_average_name:None for plate_average_name in averaged_plates_names_dict.keys()}
    for plate_average_name in averaged_plates_dict.keys():
        plates_to_average = averaged_plates_names_dict[plate_average_name]
        plates_to_average = [plates_dict[plate_name] for plate_name in plates_to_average]
        averaged_plates_dict[plate_average_name]=average_plates(plates_to_average,raw_df)
    return averaged_plates_dict

def load_plate_groups(config_path):
    xls = pd.ExcelFile(config_path)
    df = pd.read_excel(xls, 'plate_groups',index_col=0,header=None)
    replicates = df.index.tolist()[1:]
    groups = [str(group) for group in df.columns.tolist()]
    plate_groups = {replicate:{group:None for group in groups} for replicate in replicates}
    for group in groups:
        for replicate in replicates:
            #well_replicates = df.filter(like=group).loc[replicate].tolist()[0]
            plate_groups[replicate][group]=df.loc[replicate][int(group)]
    return plate_groups

def normalize_plate(plate,reference_wells,raw_df,ctrl_avg_name):
    features = get_features(raw_df)
    normalized_plate=create_well_dict(raw_df)
    normalized_plate = add_well_to_well_dict([ctrl_avg_name],normalized_plate, raw_df)
    for feature in features:
        control_values = [plate[well][feature] for well in reference_wells]
        control_avg = np.mean(np.array(control_values))
        normalized_plate[ctrl_avg_name][feature]=control_avg
        for well in normalized_plate.keys():
            if well not in ctrl_avg_name:
                try:
                    normalized_plate[well][feature] = plate[well][feature]/control_avg
                except:
                    normalized_plate[well][feature] = plate[well][feature]
    return normalized_plate


def normalize_all_plates(plates_dict,reference_wells,raw_df,ctrl_avg_name):
    normalized_plates={replicate:{} for replicate in plates_dict.keys()}
    for replicate, condition_plates in plates_dict.items():
        for condition, plate in condition_plates.items():
            normalized_plates[replicate][condition]=normalize_plate(plate,reference_wells,raw_df,ctrl_avg_name)
    return normalized_plates

def create_table_for_feature(feature,plates_dict):
    conditions = list(plates_dict.keys())
    replicates = list(list(plates_dict.values())[0].keys())
    doses=list(plates_dict[conditions[0]][replicates[0]].keys())
    col_names=[]
    for condition in conditions:
        for replicate in replicates:
            col_names.append(str(condition)+"_"+str(replicate))
    feature_table = {col_name:[] for col_name in col_names}
    for dose in doses:
        for replicate in replicates:
            for condition in conditions:
                col_name=(str(condition)+"_"+str(replicate))
                try:
                    value=plates_dict[condition][replicate][dose][feature]
                except:
                    value=None
                feature_table[col_name].append(value)
    feature_table=pd.DataFrame(feature_table)
    feature_table.columns = pd.MultiIndex.from_tuples([(c.split("_")) for c in feature_table.columns])
    feature_table.index=doses
    return feature_table

def create_feature_results_table(feature,experiment_dict):
    replicates = list(experiment_dict_values.keys())
    conditions = list(list(experiment_dict_values.values()).keys())
    col_names=[]
    for replicate in replicates:
        for condition in conditions:
            col_names.append(str(replicate)+"_"+str(condition))
    feature_table = {col_name:[] for col_name in col_names}
    for condition in conditions:
        for replicate in replicates:
            col_name=(str(condition)+"_"+str(replicate))
            for dose in doses:
                feature_table[col_name].append(plates_dict[replicate][dose][condition][feature])
    feature_table=pd.DataFrame(feature_table)
    feature_table.columns = pd.MultiIndex.from_tuples([(c.split("_")) for c in feature_table.columns])
    feature_table.index = replicates
    return feature_table

def create_all_feature_tables(plates_dict,features):
    feature_tables={feature:None for feature in features}
    for feature in features:
        feature_tables[feature]=create_table_for_feature(feature,plates_dict)
    return feature_tables

def feature_tables_to_excel(feature_tables,outpath):
    def remove_inval_chars(name):
        inval_chars=['[',']',':','*','?','/','\\']
        for char in inval_chars:
            name=name.replace(char,"")
        return name
    with pd.ExcelWriter(outpath) as writer:
        for feature in feature_tables.keys():
            feature_tables[feature].to_excel(writer, sheet_name=remove_inval_chars(feature[:31]))

def create_duplicate_wells():
    rows=[string.ascii_uppercase[i] for i in range(0,8,2)]
    cols=[i+1 for i in range(12)]
    wells = []
    for row in rows:
        for col in cols:
            wells.append(str(row)+str(col).zfill(2))
    return wells

def make_experiment_dict_locations(plate_groups,plate_layout,conditions):
    experiment_dict={condition:{} for condition in conditions}
    #experiment_dict={replicate:{} for replicate in plate_layout.keys()}
    for replicate, conditions in plate_layout.items():
        for condition,doses in conditions.items():
            experiment_dict[condition][replicate] = {dose:locations for dose,locations in doses.items()}
    return experiment_dict

def make_experiment_dict_values(plates,experiment_dict_locations,features):
    experiment_dict_values=copy.deepcopy(experiment_dict_locations)
    for condition,replicates in experiment_dict_locations.items():
        for replicate, doses in replicates.items():
            for dose,locations in doses.items():
                feature_value_dict = {feature:average_wells(locations,replicate,feature,plates,plate_groups) for feature in features}
                experiment_dict_values[condition][replicate][dose]= feature_value_dict
    return experiment_dict_values

def average_wells(locations,replicate,feature,plates,plate_groups):
    average=0
    for location in locations:
        average+=location_to_value(location,replicate,feature,plates,plate_groups)
    return average/float(len(locations))

def location_to_value(location,replicate,feature,plates,plate_groups):
    well, plate_group = location
    plate_name = plate_groups[replicate][str(plate_group)]
    value = plates[plate_name][well][feature]
    return value

def normalize_experiment(experiment_dict_values,ctrl_positions,features,plates):
    experiment_dict_values_normalized=copy.deepcopy(experiment_dict_values)
    for condition,replicates in experiment_dict_values.items():
        for replicate, doses in replicates.items():
            ctrl_positions_replicate = ctrl_positions[replicate]
            feature_control_vals={feature:average_wells(ctrl_positions_replicate,replicate,feature,plates,plate_groups) for feature in features}
            for dose,values in doses.items():
                feature_value_dict = {}
                for feature in features:
                    ctrl_value = feature_control_vals[feature]
                    if ctrl_value == 0:
                        ctrl_value = 1
                    condition_value=experiment_dict_values[condition][replicate][dose][feature]
                    feature_value_dict[feature]=condition_value/ctrl_value
                    experiment_dict_values_normalized[condition][replicate][dose]= feature_value_dict
    return experiment_dict_values_normalized

def write_values_heat_map(plates_dict,features,outpath):
    workbook = xlsxwriter.Workbook(outpath)
    with pd.ExcelWriter(outpath) as writer:
        for feature in features:
            sheet_rows=[]
            for plate in plates_dict.keys():
                sheet_rows.append([plate])
                values=[]
                for r in range(65,65+8,1):
                    values.append([])
                    row=[]
                    for c in range(12):
                        well=chr(r)+str(c+1).zfill(2)
                        row.append(plates_dict[plate][well][feature])
                    sheet_rows.append(row)
            sheet_rows.append([""])
            pd.DataFrame(sheet_rows).to_excel(writer, sheet_name=remove_inval_chars(feature[:31]))

def create_reference_wells():
    rows=[string.ascii_uppercase[i] for i in range(8)]
    cols=[i+1 for i in range(6,12)]
    wells = []
    for row in rows:
        for col in cols:
            wells.append((str(row)+str(col).zfill(2),2))
    return wells

def remove_inval_chars(name):
    inval_chars=['[',']',':','*','?','/','\\']
    for char in inval_chars:
        name=name.replace(char,"")
    return name

rows=[string.ascii_uppercase[i] for i in range(8)]
cols=[i+1 for i in range(12)]
conditions = []
for row in rows:
    for col in cols:
        conditions.append(str(row)+str(col).zfill(2))

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
if not ctrl_positions is None:
    experiment_dict_values=normalize_experiment(experiment_dict_values,ctrl_positions,features,plates_dict)
feature_tables = create_all_feature_tables(experiment_dict_values,features)
write_values_heat_map(plates_dict,features,heatmap_path)
feature_tables_to_excel(feature_tables,compiled_results_path)

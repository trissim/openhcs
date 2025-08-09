"""
Converts analysis results from Excel files into a .pzfx file format compatible with GraphPad Prism.

"""


# Imports
import pandas as pd  # For reading Excel files
import xml.sax.saxutils as saxutils  # For escaping XML special characters
from pathlib import Path  # For OS-independent file paths
from .buffer_text import FOOTER, HEADER, FOOTERMARKER  # Importing constants for footer and header XML



def parse_xlsx(file_path):
    """
    Reads all sheets from an Excel file into a dictionary of DataFrames.

    Args:
        file_path (str or Path): Path to the Excel (.xlsx) file.

    Returns:
        dict: Dictionary where keys are sheet names and values are DataFrames.
    """
    try: 
        return pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
   

def num(x):
    """
    Converts a value to a stringified float for XML.
    Returns an empty string for None or NaN.

    Args:
        x: Value to convert.

    Returns:
        str: Stringified float or empty string.
    """
    try:
        if x is None or str(x) == 'nan':
            return ''
        return str(float(x))
    except Exception:
        print(f"Warning: failed to convert '{x}' to float.")
        return ''


def get_replicate_info(df):
    """
    Infers the number of replicate columns per group by inspecting the first row of data.

    Args:
        df (pd.DataFrame): DataFrame after header rows are removed.

    Returns:
        int: Number of replicates detected (e.g., 3 for N1, N2, N3).
    """
    labels = df.iloc[0, 1:]  
    seen = set()
    replicates = 0
    for label in labels:
        if not isinstance(label, str):
            break
        label = label.strip()
        if label in seen:
            break  # Looped back to the start of a new group
        seen.add(label)
        replicates += 1
    if replicates == 0:
        print("Warning: No replicate columns detected. Defaulting to 1.")
        return 1
    return replicates


def build_x_columns(units, x_vals, x_column_width=81, x_decimals=1):
    """
    Builds the XML for the X columns.

    Args:
        x_vals (list): List of X values.
        x_column_width (int): Width for X columns.
        x_decimals (int): Decimals for X columns.

    Returns:
        str: XML string for X columns.
    """
    
    # X column
    xml = f'  <XColumn Width="{x_column_width}" Subcolumns="1" Decimals="{x_decimals}">\n    <Title>[{units}]</Title>\n    <Subcolumn>\n'
    for x in x_vals:
        xml += f'      <d>{num(x)}</d>\n'
    xml += '    </Subcolumn>\n  </XColumn>\n'
    
    # X advanced column
    xml += f'  <XAdvancedColumn Version="1" Width="{x_column_width}" Decimals="{x_decimals}" Subcolumns="1">\n    <Title>[{units}]</Title>\n    <Subcolumn>\n'
    for x in x_vals:
        xml += f'      <d>{num(x)}</d>\n'
    xml += '    </Subcolumn>\n  </XAdvancedColumn>\n'
    
    return xml

def build_y_columns(df, replicates, y_widths=None, y_width_default=390, y_decimals=14):
    """
    Builds the XML for the Y columns (replicates).

    Args:
        df (pd.DataFrame): DataFrame with data (header rows removed).
        y_groups (list): List of lists, each containing column names for a group.
        replicates (int): Number of replicates per group.
        y_widths (list, optional): List of widths for each Y group.
        y_width_default (int, optional): Default width for Y columns.
        y_decimals (int, optional): Decimals for Y columns.

    Returns:
        str: XML string for Y columns.
    """
    y_cols = df.columns[1:]
    y_data = df.iloc[:, 1:]
    # Group Y columns by replicates (e.g., 3 columns per group)
    y_groups = [y_cols[i:i+replicates] for i in range(0, len(y_cols), replicates)]
    if y_widths is None:
        y_widths = []
    
    xml = ''
    # Y columns (replicates)
    for idx, group in enumerate(y_groups):
        sample_name = group[0]
        width = y_widths[idx] if idx < len(y_widths) else y_width_default
        xml += f'  <YColumn Width="{width}" Decimals="{y_decimals}" Subcolumns="{replicates}">\n    <Title>{saxutils.escape(str(sample_name))}</Title>\n'
        for col in group:
            xml += '    <Subcolumn>\n'
            for val in y_data[col].tolist():
                xml += f'      <d>{num(val)}</d>\n'
            xml += '    </Subcolumn>\n'
        xml += '  </YColumn>\n'
    return xml

def df_to_table1024(
    df, table_id, title, units, y_width_default=390, y_widths=None,
    x_column_width=81, x_decimals=1, y_decimals=14
):
    """
    Converts a DataFrame to a Prism Table1024 XML block.
    """

    df = df.copy()
    # Infer replicates 
    replicates = get_replicate_info(df)
    
    # Remove header rows
    df = df.iloc[2:].reset_index(drop=True)
    x_vals = df.iloc[:, 0].dropna().tolist()
    
    # Start XML for this table
    xml = f'<Table1024 ID="{table_id}" XFormat="numbers" YFormat="replicates" Replicates="{replicates}" TableType="XY" EVFormat="AsteriskAfterNumber">\n'
    xml += f'  <Title>{saxutils.escape(title)}</Title>\n'
    
    # X columns
    xml += build_x_columns(units, x_vals, x_column_width, x_decimals)
    
    # Y columns
    xml += build_y_columns(df, replicates, y_widths, y_width_default, y_decimals)
    
    xml += '</Table1024>\n'
    return xml
    
    
def insert_tablesequence(table_ids):
    """
    Generates the TableSequence XML block.
    Args:
        table_ids (list): List of table IDs.
    Returns:
        str: XML block for TableSequence.
    """
    seq = "<TableSequence>\n"
    for i, tid in enumerate(table_ids):
        if i == len(table_ids) - 1:
            seq += f'<Ref ID="{tid}" Selected="1"/>\n'
        else:
            seq += f'<Ref ID="{tid}"/>\n'
    seq += "</TableSequence>\n"
    return seq

   
    
def parse_footer(analysis_path):
    """    
    Parses the footer from the analysis template.
    
    Args:
        analysis_path (str or Path): Path to the analysis template file.
    Returns:
        str: Footer XML block or default footer if not found.
    """
    
    #file handling
    if analysis_path is None:
        return FOOTER
    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            template = f.read()
            index = template.find(FOOTERMARKER)
            if index == -1:
                print("Warning: Initial footer sequence not found in analysis template. Appending default footer.")
                return FOOTER
            else:
                return template[index:]
    except FileNotFoundError:
        print(f"Warning: Footer file '{analysis_path}' not found. Appending default footer.")
        return FOOTER

def write_pzfx(data_dict, units = "uM", analysis_path=None):
    """
    Assembles the full PZFX XML.

    Args:
        data_dict (dict): Dictionary of {sheet_name: DataFrame}.
        output_file (str or Path): Path to write the .pzfx file.

    Returns:
        None
    """
    #Header
    xml = HEADER
    
    #Tables
    table_ids = []
    for i, (sheet_name, df) in enumerate(data_dict.items()):
        table_id = f"Table{i+1}"
        table_ids.append(table_id)
        xml += df_to_table1024(df, table_id, sheet_name, units)
    
    #TableSequence
    xml = xml.replace("<!--TABLESEQUENCE-->", insert_tablesequence(table_ids))
    
    #Footer
    xml += parse_footer(analysis_path)
   
    return xml


def convertFile(input_file, output_file, analysis_path = None):
    """
    Main conversion function.
    Checks for required files, loads data and metadata, and writes the PZFX file.

    Args:
        input_file (str or Path): Path to the input Excel file.
        metadata_path (str or Path): Path to the metadata .ini file.

    Returns:
        nothing
    """
    #parse data
    data_dict = parse_xlsx(input_file)
    if data_dict is None:
        print("Error: Failed to parse the input Excel file. Exiting.")
        return
    xml = write_pzfx(data_dict, analysis_path)
    
    #output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml)
        print(f"Conversion complete. Output written to {output_file}")
    

if __name__ == "__main__":
    # Set up file paths for input Excel and output PZFX
    input_file = str(Path.cwd().joinpath('compiled_results_normalized.xlsx'))  # Path to your .xlsx file
    
    #conversion without analysis template
    output_file = str(input_file).replace(".xlsx", ".pzfx") 
    convertFile(input_file, output_file)
    
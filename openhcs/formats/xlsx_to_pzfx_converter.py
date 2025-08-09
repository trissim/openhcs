"""
Converts analysis results from Excel files into a .pzfx file format compatible with GraphPad Prism.

"""


# Imports
import pandas as pd  # For reading Excel files
import xml.sax.saxutils as saxutils  # For escaping XML special characters
from pathlib import Path  # For OS-independent file paths
import re 

HEADER = '''<?xml version="1.0" encoding="UTF-8"?>
<GraphPadPrismFile xmlns="http://graphpad.com/prism/Prism.htm" PrismXMLVersion="5.00">
<Created>
<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="9.3.1.471" Login="epic" DateTime="2025-08-06T00:26:07.618398"/>
<MostRecentVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="10.5.0.774" Login="jatha" DateTime="2025-08-06T12:40:05-05:00"/>
</Created>


<!--TABLESEQUENCE-->
'''

FOOTER = f"""<!--Analyses, graphs and layouts as compressed binary. Don't edit this part of the file.-->

<Template xmlns:dt="urn:schemas-microsoft-com:datatypes" dt:dt="bin.base64">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
sQOCPU6qC/bYLsEeEuwhwR4S7CHBHhLsIcEeEuwhwR4S7CHBHhLsIcEeEuwhwR4S7CHBHhLs
IcEeEuwhwR4S7CHBHhLsIcEeEuwhwR4S7CHBHhLsIcEeEuwhwR4S7CHBHhLsIcEeEuwhwR4S
7CHBHhLsIcEeEuwhwR4S7CHBHhLsIcEeEuyRIMEeO5WCPQ4oBXvsUgr22K0U7LFXKdjjoFKw
xz7LXoFRjtGPBHvst71XwIml2d3LfFb6Qtt7mc8+NbO9VyPTAdrey3wyuLntvcxsj4tt73W8
6Txt73WC6VBt72U+hX2Z7b3MZ98ut7FX8HHy7Zbn/O+IE/qyw3noi3m4HQka+rIjzlBZQ192
6Al9sQ6XztCXnfWUg0ECO3SFvlgL0hn6csBRQUqhL7HmR0foyy4t82Mz9EX//ERCX3Y7Lkgh
9EV/QZHQl71aZshm6Iv+giKXhIOOClIJfbFWozP0ZV/oDBsdZTVJzpJSot3adIW+7E+NZBYk
eCkXpEUevz+aUjw/tlIiC4ILj51Smh07pVx07JTS/Ngp5eJjp5RLjp1SLj12Srns2Cnl8mOh
lHgRb2ZB4Yi3/snuRbyhxd0JFe6EWK6dsFzaCbFcOyGWaydEfaEt2wWau0BzF2juAs1doLkL
NNEZ7QbN3aC5GzR3g+Zu0NwNmmhO9oLmXtDcC5p7QXMvaO4FzX2guQ8094HmPtDcB5r7QHMf
aO4Hzf2guR8094PmftDcD5r7QRM/CTgAmgdA8wBoHgDNA6B5ADTRGx0EzYOgeRA0D4LmwSgR
b9mhiLdbfNl5kPDW2leQ0cqXmxuOepOIt3qbRLz9f4p4C/yoWuizNicRb4cg4i0YtWVGvP0g
EW+GRLxJxJtEvEnEm0S8ScSbRLxJxJtEvEnEm0S8ScSbRLxJxJtEvEnEm0S8ScSbRLxJxJtE
vEnEm0S8ScSbRLxJxJtEvEnEm0S8ScSbRLxJxJtEvEnEm0S8ScSbRLxJxJtEvEnEm0S8ScSb
RLxJxJtEvEnEm0S8ScSbRLwlRsTbYaWIt0ZpKhFvRppKxJsnTSXiLS1NJeLt5DSViLeG4b2+
z6x7NihWPxLxlm5jr+Ajwz9YkluyjdgxXoecx3iZhzuUoDFeh+IMlTXG65CeGC/rcOmM8Tpc
Tzn4sPghXTFe1oJ0xng1SnNSkFKMV6z50RHjZaTpmB+bMV765ycS4+VxXJBCjJf+giIxXmla
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
7w==</Template></GraphPadPrismFile>"""


def parse_xlsx(file_path):
    """
    Reads all sheets from an Excel file into a dictionary of DataFrames.

    Args:
        file_path (str or Path): Path to the Excel (.xlsx) file.

    Returns:
        dict: Dictionary where keys are sheet names and values are DataFrames.
    """
    try: 
        all_sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
   
    return all_sheets

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
    import math

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
    
    initialFooterSequence =  """<!--Analyses, graphs and layouts as compressed binary. Don't edit this part of the file.-->"""
    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            template = f.read()
            index = template.find(initialFooterSequence)
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
    xml.replace("<!--TABLESEQUENCE-->", insert_tablesequence(table_ids))
    
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
    
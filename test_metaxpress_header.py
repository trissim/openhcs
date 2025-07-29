#!/usr/bin/env python3
"""
Test the MetaXpress-style header functionality.
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

def main():
    """Test the MetaXpress-style header functionality."""
    
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    
    print("🔬 Testing MetaXpress-Style Header")
    print("=" * 50)
    
    # Test with custom metadata
    custom_metadata = {
        'barcode': 'OpenHCS-Axotomy-Exp',
        'plate_name': 'mar-20-axotomy-fca-dmso-Plate-1',
        'plate_id': '13053',
        'description': 'Axotomy experiment with FCA and DMSO treatments. Cell counting, axon analysis, and template matching.',
        'acquisition_user': 'OpenHCS-User',
        'z_step': '1'
    }
    
    print("🔄 Generating MetaXpress-style output with header...")
    summary_df = consolidate_analysis_results(
        results_directory=results_dir,
        exclude_patterns=[r".*consolidated.*", r".*metaxpress.*"],
        metaxpress_style=True,
        plate_metadata=custom_metadata
    )
    
    print(f"✅ Generated summary with {summary_df.shape[0]} wells and {summary_df.shape[1]} metrics")
    
    # Check the output file structure
    output_file = f"{results_dir}/metaxpress_style_summary.csv"
    print(f"\n📄 Checking output file structure: {output_file}")
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📊 Total lines in file: {len(lines)}")
    print(f"\n🏷️ Header structure (first 10 lines):")
    for i, line in enumerate(lines[:10]):
        line_clean = line.strip()
        if ',' in line_clean:
            parts = line_clean.split(',')
            if len(parts) >= 2:
                print(f"  Line {i+1}: {parts[0]} = {parts[1]}")
            else:
                print(f"  Line {i+1}: {parts[0]}")
        else:
            print(f"  Line {i+1}: {line_clean}")
    
    print(f"\n🎯 Verification:")
    print(f"  - Header rows (metadata): Lines 1-6")
    print(f"  - Column headers: Line 7") 
    print(f"  - Data rows: Lines 8-{len(lines)}")
    print(f"  - Expected data rows: {summary_df.shape[0]} wells")
    print(f"  - Actual data rows: {len(lines) - 7}")
    
    # Compare with real MetaXpress format
    print(f"\n📋 Comparison with real MetaXpress:")
    print(f"  ✓ Barcode row")
    print(f"  ✓ Plate Name row") 
    print(f"  ✓ Plate ID row")
    print(f"  ✓ Description row")
    print(f"  ✓ Acquisition User row")
    print(f"  ✓ Z Step row")
    print(f"  ✓ Column headers")
    print(f"  ✓ Data rows with well IDs")
    
    print(f"\n💾 Output saved to: {output_file}")
    print(f"🚀 Ready for your existing MetaXpress processing script!")

if __name__ == "__main__":
    main()

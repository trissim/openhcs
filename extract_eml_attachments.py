#!/usr/bin/env python3
"""
Extract Attachments from EML Files

This script processes a folder containing .eml files (email message files) and
extracts all attachments to a specified output directory.

Usage:
    python extract_eml_attachments.py input_folder output_folder
    
Example:
    python extract_eml_attachments.py ./eml_files ./extracted_attachments
"""

import email
import os
import sys
import argparse
from pathlib import Path
from email.mime.multipart import MIMEMultipart
import mimetypes

def sanitize_filename(filename):
    """
    Sanitize filename to remove invalid characters for filesystem.
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_attachment"
    
    return filename

def extract_attachments_from_eml(eml_file_path, output_dir, create_subfolders=True):
    """
    Extract all attachments from a single .eml file.
    
    Args:
        eml_file_path: Path to the .eml file
        output_dir: Directory to save attachments
        create_subfolders: If True, create subfolder for each email
    
    Returns:
        List of extracted attachment filenames
    """
    extracted_files = []
    
    try:
        # Read the .eml file
        with open(eml_file_path, 'rb') as f:
            msg = email.message_from_bytes(f.read())
        
        # Create subfolder for this email if requested
        if create_subfolders:
            eml_name = Path(eml_file_path).stem
            email_output_dir = Path(output_dir) / sanitize_filename(eml_name)
        else:
            email_output_dir = Path(output_dir)
        
        # Create output directory
        email_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract email metadata for reference
        subject = msg.get('Subject', 'No Subject')
        sender = msg.get('From', 'Unknown Sender')
        date = msg.get('Date', 'Unknown Date')
        
        print(f"\n📧 Processing: {Path(eml_file_path).name}")
        print(f"   Subject: {subject}")
        print(f"   From: {sender}")
        print(f"   Date: {date}")
        
        # Counter for unnamed attachments
        unnamed_counter = 1
        
        # Extract attachments
        for part in msg.walk():
            # Skip multipart containers
            if part.get_content_maintype() == 'multipart':
                continue
            
            # Skip if no content disposition (likely email body)
            content_disposition = part.get('Content-Disposition')
            if content_disposition is None:
                continue
            
            # Check if it's an attachment
            if 'attachment' not in content_disposition.lower():
                continue
            
            # Get original filename to extract extension
            original_filename = part.get_filename()

            # Extract file extension
            if original_filename:
                _, extension = os.path.splitext(original_filename)
            else:
                # Try to guess extension from content type
                content_type = part.get_content_type()
                extension = mimetypes.guess_extension(content_type) or '.bin'

            # Create new filename based on parent folder name
            if create_subfolders:
                # Use the email folder name as the base filename
                base_name = sanitize_filename(eml_name)
                if len(extracted_files) > 0:  # Multiple attachments in same email
                    filename = f"{base_name}_attachment_{len(extracted_files) + 1}{extension}"
                else:
                    filename = f"{base_name}{extension}"
            else:
                # When not using subfolders, include email name in filename
                base_name = sanitize_filename(eml_name)
                if len(extracted_files) > 0:  # Multiple attachments in same email
                    filename = f"{base_name}_attachment_{len(extracted_files) + 1}{extension}"
                else:
                    filename = f"{base_name}{extension}"
            
            # Handle duplicate filenames
            original_filename = filename
            counter = 1
            while (email_output_dir / filename).exists():
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1
            
            # Save attachment
            filepath = email_output_dir / filename
            
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    with open(filepath, 'wb') as f:
                        f.write(payload)
                    
                    file_size = len(payload)
                    print(f"   ✅ Extracted: {filename} ({file_size:,} bytes)")
                    extracted_files.append(str(filepath))
                else:
                    print(f"   ⚠️  Warning: Empty payload for {filename}")
                    
            except Exception as e:
                print(f"   ❌ Error extracting {filename}: {e}")
        
        if not extracted_files:
            print(f"   📭 No attachments found")
            
    except Exception as e:
        print(f"   ❌ Error processing {eml_file_path}: {e}")
    
    return extracted_files

def process_eml_folder(input_folder, output_folder, create_subfolders=True, name_filter=None):
    """
    Process all .eml files in a folder and extract attachments.

    Args:
        input_folder: Folder containing .eml files
        output_folder: Folder to save extracted attachments
        create_subfolders: Create subfolder for each email
        name_filter: Only process files containing this string in filename
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"❌ Input folder does not exist: {input_folder}")
        return
    
    # Find all .eml files
    eml_files = list(input_path.glob("*.eml"))

    # Apply name filter if specified
    if name_filter:
        eml_files = [f for f in eml_files if name_filter.lower() in f.name.lower()]
        print(f"🔍 Filtering for files containing: '{name_filter}'")

    if not eml_files:
        filter_msg = f" matching '{name_filter}'" if name_filter else ""
        print(f"📭 No .eml files{filter_msg} found in {input_folder}")
        return
    
    print(f"🔍 Found {len(eml_files)} .eml files")
    print(f"📁 Output directory: {output_folder}")
    print(f"📂 Create subfolders: {create_subfolders}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    total_attachments = 0
    processed_emails = 0
    
    for eml_file in eml_files:
        extracted = extract_attachments_from_eml(eml_file, output_path, create_subfolders)
        total_attachments += len(extracted)
        if extracted:
            processed_emails += 1
    
    print(f"\n🎉 Processing complete!")
    print(f"   📧 Emails processed: {len(eml_files)}")
    print(f"   📧 Emails with attachments: {processed_emails}")
    print(f"   📎 Total attachments extracted: {total_attachments}")
    print(f"   📁 Output location: {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract attachments from .eml files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_eml_attachments.py ./eml_files ./attachments
  python extract_eml_attachments.py ./emails ./output --no-subfolders
        """
    )
    
    parser.add_argument('input_folder', help='Folder containing .eml files')
    parser.add_argument('output_folder', help='Folder to save extracted attachments')
    parser.add_argument('--no-subfolders', action='store_true',
                       help='Do not create subfolders for each email')
    parser.add_argument('--filter', type=str,
                       help='Only process files containing this string in filename')
    
    args = parser.parse_args()
    
    create_subfolders = not args.no_subfolders

    print("📧 EML Attachment Extractor")
    print("=" * 50)

    process_eml_folder(args.input_folder, args.output_folder, create_subfolders, args.filter)

if __name__ == "__main__":
    main()

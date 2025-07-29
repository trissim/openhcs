#!/usr/bin/env python3
"""
Quick PDF inspection script to understand the format of invoices and statements.
"""

import pdfplumber
import re
from pathlib import Path

def inspect_pdf(pdf_path, max_lines=50):
    """Extract and display first few lines of PDF text."""
    print(f"\n{'='*60}")
    print(f"📄 INSPECTING: {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages[:2]:  # First 2 pages only
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            lines = text.split('\n')
            for i, line in enumerate(lines[:max_lines]):
                if line.strip():
                    print(f"{i+1:3d}: {line}")
                    
            if len(lines) > max_lines:
                print(f"... ({len(lines) - max_lines} more lines)")
                
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")

def find_augment_transactions(pdf_path):
    """Look for Augment Code transactions in statement."""
    print(f"\n🔍 SEARCHING FOR AUGMENT TRANSACTIONS: {Path(pdf_path).name}")
    print("-" * 60)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            lines = text.split('\n')
            augment_lines = []
            
            for i, line in enumerate(lines):
                if 'augment' in line.lower():
                    # Include context lines
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    context = lines[start:end]
                    augment_lines.append((i, context))
            
            if augment_lines:
                for line_num, context in augment_lines:
                    print(f"\nFound at line {line_num}:")
                    for j, ctx_line in enumerate(context):
                        marker = ">>> " if j == 2 else "    "
                        print(f"{marker}{ctx_line}")
            else:
                print("No Augment transactions found")
                
    except Exception as e:
        print(f"❌ Error searching {pdf_path}: {e}")

def main():
    print("🔍 PDF Content Inspector")
    print("=" * 60)
    
    # Inspect one invoice PDF
    invoice_dir = "/home/ts/Downloads/extracted_attachments_renamed"
    invoice_files = list(Path(invoice_dir).rglob("*.pdf"))
    
    if invoice_files:
        print("\n📧 SAMPLE INVOICE PDF:")
        inspect_pdf(invoice_files[0], max_lines=30)
    
    # Inspect statement PDFs
    statement_dir = "/home/ts/Downloads/temp_statements"
    
    # Bank statements
    bank_files = list(Path(statement_dir).glob("bank_*.pdf"))
    if bank_files:
        print("\n🏦 SAMPLE BANK STATEMENT:")
        inspect_pdf(bank_files[0], max_lines=40)
        find_augment_transactions(bank_files[0])
    
    # Credit statements  
    credit_files = list(Path(statement_dir).glob("credit_*.pdf"))
    if credit_files:
        print("\n💳 SAMPLE CREDIT STATEMENT:")
        inspect_pdf(credit_files[0], max_lines=40)
        find_augment_transactions(credit_files[0])

if __name__ == "__main__":
    main()

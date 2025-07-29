#!/usr/bin/env python3
"""
Financial Reconciliation Script

This script processes Augment and OpenRouter invoices and bank/credit statements
to create a comprehensive reconciliation CSV report.

Usage:
    python financial_reconciliation.py
"""

import os
import zipfile
import csv
import re
import pdfplumber
import email
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import calendar
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
import os

# Try to import PDF manipulation libraries
try:
    import fitz  # PyMuPDF
    PDF_HIGHLIGHT_AVAILABLE = True
except ImportError:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.colors import yellow, red
        PDF_HIGHLIGHT_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: PDF highlighting not available. Install PyMuPDF or reportlab:")
        print("   pip install PyMuPDF")
        print("   or")
        print("   pip install reportlab")
        PDF_HIGHLIGHT_AVAILABLE = False

class InvoiceData:
    def __init__(self, invoice_number: str, date: str, amount_usd: float, file_path: str):
        self.invoice_number = invoice_number
        self.date = date
        self.amount_usd = amount_usd
        self.file_path = file_path

    def __repr__(self):
        return f"Invoice({self.invoice_number}, {self.date}, ${self.amount_usd})"

class StatementTransaction:
    _next_id = 0  # Class variable for unique IDs

    def __init__(self, date: str, amount_usd: float, description: str, statement_file: str, statement_type: str, amount_cad: float = None):
        self.date = date
        self.amount_usd = amount_usd
        self.amount_cad = amount_cad
        self.description = description
        self.statement_file = statement_file
        self.statement_type = statement_type

        self.transaction_id = StatementTransaction._next_id  # Unique ID for tracking
        StatementTransaction._next_id += 1

    def __repr__(self):
        return f"Transaction({self.date}, ${self.amount_usd}, {self.statement_type})"

def extract_zip_with_prefix(zip_path: str, extract_dir: str, prefix: str) -> List[str]:
    """
    Extract ZIP file and rename PDFs with prefix to avoid naming conflicts.
    
    Returns list of extracted file paths.
    """
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if file_info.filename.endswith('.pdf'):
                # Extract to temporary location
                zip_ref.extract(file_info, extract_dir)
                
                # Rename with prefix
                original_path = Path(extract_dir) / file_info.filename
                new_name = f"{prefix}_{file_info.filename}"
                new_path = Path(extract_dir) / new_name
                
                original_path.rename(new_path)
                extracted_files.append(str(new_path))
                
                print(f"   📄 Extracted: {new_name}")
    
    return extracted_files

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"   ⚠️  Warning: Could not extract text from {pdf_path}: {e}")
        return ""

def parse_invoice_pdf(pdf_path: str) -> Optional[InvoiceData]:
    """Parse invoice PDF to extract invoice number, date, and USD amount."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None

    # Extract invoice number from filename (more reliable)
    filename = Path(pdf_path).name

    # Try different invoice number patterns
    invoice_match = re.search(r'EQCIET-\d+', filename)  # Standard pattern: EQCIET-00013
    if not invoice_match:
        invoice_match = re.search(r'Invoice-([A-F0-9-]+)', filename)  # Alternative pattern: Invoice-9A7F765C-0002

    invoice_number = invoice_match.group() if invoice_match else "Unknown"

    # Look for invoice date in the text
    # Pattern: "Invoice Date May 26, 2025"
    date_match = re.search(r'Invoice Date\s+(\w+ \d{1,2}, \d{4})', text)
    if not date_match:
        # Alternative pattern: "May 26, 2025" standalone
        date_match = re.search(r'(\w+ \d{1,2}, \d{4})', text)

    date = date_match.group(1) if date_match else None

    # Look for USD amount - "Amount due $25.00"
    amount_match = re.search(r'Amount due \$(\d+\.?\d*)', text)
    if not amount_match:
        # Alternative patterns
        amount_match = re.search(r'Total.*?\$(\d+\.?\d*)', text)

    amount_usd = float(amount_match.group(1)) if amount_match else None

    if invoice_number and date and amount_usd:
        return InvoiceData(invoice_number, date, amount_usd, pdf_path)

    print(f"   ⚠️  Could not parse invoice: {filename} (date: {date}, amount: {amount_usd})")
    return None

def parse_openrouter_eml(eml_path: str) -> Optional[InvoiceData]:
    """Parse OpenRouter .eml file to extract invoice number, date, and USD amount."""
    try:
        with open(eml_path, 'rb') as f:
            msg = email.message_from_bytes(f.read())

        # Extract receipt number from subject
        subject = msg.get('Subject', '')
        receipt_match = re.search(r'#(\d{4}-\d{4})', subject)
        invoice_number = receipt_match.group(1) if receipt_match else "Unknown"

        # Get email body (prefer HTML part for OpenRouter)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_content = payload.decode('utf-8', errors='ignore')
                        # Extract text from HTML
                        body = re.sub(r'<[^>]+>', ' ', html_content)
                        body = re.sub(r'\s+', ' ', body)
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')

        # Look for the receipt line: "Receipt from OpenRouter, Inc [#1429-6053] Amount paid $10.90 Date paid May 20, 2025, 9:27:22 AM"
        receipt_pattern = r'Receipt from OpenRouter[^#]*#(\d{4}-\d{4})[^$]*Amount paid \$(\d+\.?\d*)[^D]*Date paid ([^,]+)'
        receipt_match = re.search(receipt_pattern, body)

        if receipt_match:
            receipt_num, amount_str, date_str = receipt_match.groups()

            # Verify receipt number matches subject
            if receipt_num == invoice_number:
                amount_usd = float(amount_str)
                date = date_str.strip()

                return InvoiceData(f"OR-{invoice_number}", date, amount_usd, eml_path)

        print(f"   ⚠️  Could not parse OpenRouter email: {Path(eml_path).name}")
        return None

    except Exception as e:
        print(f"   ❌ Error parsing OpenRouter email {Path(eml_path).name}: {e}")
        return None

def parse_statement_pdf(pdf_path: str, statement_type: str) -> List[StatementTransaction]:
    """Parse bank/credit statement PDF to find Augment Code transactions."""
    # Extract text from ALL pages, not just the first page
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
    except Exception as e:
        print(f"   ⚠️  Warning: Could not extract text from {pdf_path}: {e}")
        return []

    if not all_text:
        return []

    text = all_text

    transactions = []
    lines = text.split('\n')


    # Extract month/year from filename for date context
    filename = Path(pdf_path).name
    # Handle filenames like "bank_June 2025 e-statement.pdf"
    month_year_match = re.search(r'(?:bank_|credit_)?(\w+) (\d{4})', filename)
    if month_year_match:
        month_name, year = month_year_match.groups()
        year = int(year)
        try:
            month_num = list(calendar.month_name).index(month_name)
        except ValueError:
            year, month_num = 2025, 1  # fallback if month name not found
    else:
        year, month_num = 2025, 1  # fallback

    augment_lines_found = 0
    transaction_index = 0  # Sequential index for detected transactions
    for i, line in enumerate(lines):
        if 'augment' in line.lower():
            augment_lines_found += 1

        # Look for both Augment and OpenRouter transactions
        line_lower = line.lower()
        is_augment = ('augment code' in line_lower or 'augmentcode' in line_lower)
        is_openrouter = 'openrouter' in line_lower

        if is_augment or is_openrouter:
            amount_usd = None
            amount_cad = None
            date = None

            # Bank format: "Opos25.00AugmentCode+1408357014CAUS" or "Opos10.90OpenRouter,Inc+1848297448NV US"
            if statement_type == "bank":
                # Extract USD amount from current line
                amount_match = re.search(r'Opos(\d+\.?\d*)', line)
                if amount_match:
                    amount_usd = float(amount_match.group(1))

                # Extract date and CAD amount from immediately preceding line ONLY
                if i > 0:
                    prev_line = lines[i-1]
                    # Pattern: "Jun2 Pointofsalepurchase 35.44 253.16" or "Apr11 Pointofsalepurchase 15.76 1,644.37"
                    date_cad_match = re.search(r'(\w{3})(\d{1,2})\s*.*?(\d+\.\d{2})\s+[\d,]+\.\d{2}', prev_line)
                    if date_cad_match:
                        month_abbr, day, cad_amount = date_cad_match.groups()
                        amount_cad = float(cad_amount)
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        if month_abbr in month_map:
                            date = f"{month_abbr} {day}, {year}"

            # Credit format: "001 Jun 9 Jun 10 AUGMENT CODE 35.13" + "AMT 25.00 USD" on next lines
            elif statement_type == "credit":
                # Extract date and CAD amount from current line
                if is_augment:
                    # Pattern: "001 Jun 9 Jun 10 AUGMENT CODE 35.13"
                    date_cad_match = re.search(r'\d+\s+(\w{3})\s+(\d{1,2})\s+\w{3}\s+\d{1,2}\s+AUGMENT CODE\s+(\d+\.\d{2})', line)
                    if date_cad_match:
                        month_abbr, day, cad_amount = date_cad_match.groups()
                        amount_cad = float(cad_amount)
                elif is_openrouter:
                    # Pattern: "001 May 17 May 17 OPENROUTER 15.64" (if it exists)
                    date_cad_match = re.search(r'\d+\s+(\w{3})\s+(\d{1,2})\s+\w{3}\s+\d{1,2}\s+OPENROUTER\s+(\d+\.\d{2})', line)
                    if date_cad_match:
                        month_abbr, day, cad_amount = date_cad_match.groups()
                        amount_cad = float(cad_amount)

                if 'date_cad_match' in locals() and date_cad_match:
                    month_map = {
                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                    }
                    if month_abbr in month_map:
                        date = f"{month_abbr} {day}, {year}"

                # Look for USD amount in next few lines: "AMT 25.00 USD"
                search_lines = lines[i:min(len(lines), i+5)]
                for search_line in search_lines:
                    amount_match = re.search(r'AMT\s+(\d+\.?\d*)\s+USD', search_line)
                    if amount_match:
                        amount_usd = float(amount_match.group(1))
                        break

            if date and amount_usd:
                transaction = StatementTransaction(
                    date=date,
                    amount_usd=amount_usd,
                    amount_cad=amount_cad,
                    description=line.strip(),
                    statement_file=Path(pdf_path).name,
                    statement_type=statement_type
                )
                transaction.sequential_index = transaction_index  # Store sequential index directly
                transactions.append(transaction)
                transaction_index += 1  # Increment for next detected transaction
    return transactions

def normalize_date(date_str: str, year: int = 2025) -> Optional[str]:
    """
    Normalize various date formats to YYYY-MM-DD.
    """
    try:
        # Try different parsing approaches
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str

        if re.match(r'\d{1,2}/\d{1,2}', date_str):
            month, day = date_str.split('/')
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # Handle "May 26, 2025" format (from invoices) - full month names
        if re.match(r'\w+ \d{1,2}, \d{4}', date_str):
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            parts = date_str.replace(',', '').split()
            month_name = parts[0].lower()
            day = int(parts[1])
            year_val = int(parts[2])

            month_num = month_names.get(month_name)
            if month_num:
                return f"{year_val}-{month_num:02d}-{day:02d}"

        # Handle "Apr 11, 2025" format (abbreviated month names with year)
        if re.match(r'\w{3} \d{1,2}, \d{4}', date_str):
            month_names = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            parts = date_str.replace(',', '').split()
            month_abbr = parts[0].lower()
            day = int(parts[1])
            year_val = int(parts[2])

            month_num = month_names.get(month_abbr)
            if month_num:
                return f"{year_val}-{month_num:02d}-{day:02d}"

        # Handle "May 26" format (from statements)
        if re.match(r'\w{3}\s+\d{1,2}', date_str):
            month_names = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            parts = date_str.lower().split()
            month_num = month_names.get(parts[0][:3])
            if month_num:
                day = int(parts[1])
                return f"{year}-{month_num:02d}-{day:02d}"

    except Exception:
        pass

    return None

def convert_eml_to_pdf(eml_path: str, pdf_path: str) -> bool:
    """Convert EML file to PDF format using wkhtmltopdf."""
    try:
        import subprocess
        import tempfile

        # Read the EML file
        with open(eml_path, 'rb') as f:
            msg = email.message_from_bytes(f.read())

        # Extract email content
        subject = msg.get('Subject', 'No Subject')
        from_addr = msg.get('From', 'Unknown Sender')
        to_addr = msg.get('To', 'Unknown Recipient')
        date = msg.get('Date', 'Unknown Date')

        # Get email body (prefer HTML for better formatting)
        html_body = ""
        text_body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif part.get_content_type() == "text/plain":
                    text_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            if msg.get_content_type() == "text/html":
                html_body = content
            else:
                text_body = content

        # Create HTML content with email headers
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{subject}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .email-header {{ border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }}
                .email-header h2 {{ margin: 0; color: #333; }}
                .email-meta {{ color: #666; font-size: 14px; margin: 5px 0; }}
                .email-body {{ line-height: 1.6; }}
            </style>
        </head>
        <body>
            <div class="email-header">
                <h2>{subject}</h2>
                <div class="email-meta"><strong>From:</strong> {from_addr}</div>
                <div class="email-meta"><strong>To:</strong> {to_addr}</div>
                <div class="email-meta"><strong>Date:</strong> {date}</div>
            </div>
            <div class="email-body">
                {html_body if html_body else f'<pre>{text_body}</pre>'}
            </div>
        </body>
        </html>
        """

        # Write HTML to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
            temp_html.write(html_content)
            temp_html_path = temp_html.name

        try:
            # Use wkhtmltopdf to convert HTML to PDF
            result = subprocess.run([
                'wkhtmltopdf',
                '--page-size', 'A4',
                '--margin-top', '20mm',
                '--margin-bottom', '20mm',
                '--margin-left', '15mm',
                '--margin-right', '15mm',
                '--quiet',
                temp_html_path,
                pdf_path
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return True
            else:
                print(f"   ❌ wkhtmltopdf error: {result.stderr}")
                return False

        finally:
            # Clean up temporary file
            os.unlink(temp_html_path)

    except Exception as e:
        print(f"   ❌ Error converting {eml_path} to PDF: {e}")
        return False

def extract_attachments_from_emls(eml_dir: str, output_dir: str) -> int:
    """Extract PDF attachments from EML files and save them with proper names."""
    import email
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    print(f"📎 Extracting attachments from EML files...")

    for eml_file in Path(eml_dir).glob("*.eml"):
        try:
            with open(eml_file, 'rb') as f:
                msg = email.message_from_bytes(f.read())

            # Extract PDF attachments
            for part in msg.walk():
                if part.get_content_type() == 'application/pdf':
                    filename = part.get_filename()
                    payload = part.get_payload(decode=True)

                    if payload and filename:
                        # Determine provider and create proper filename
                        if 'augment' in eml_file.name.lower():
                            output_filename = f"Augment_{filename}"
                        elif 'openrouter' in eml_file.name.lower():
                            output_filename = f"OpenRouter_{filename}"
                        else:
                            output_filename = filename

                        output_path = Path(output_dir) / output_filename

                        # Save the PDF
                        with open(output_path, 'wb') as pdf_file:
                            pdf_file.write(payload)

                        extracted_count += 1
                        print(f"   ✅ Extracted: {output_filename}")

        except Exception as e:
            print(f"   ❌ Error processing {eml_file.name}: {e}")

    print(f"   📊 Total attachments extracted: {extracted_count}")
    return extracted_count

def convert_emls_to_pdfs_parallel(eml_pdf_pairs: List[Tuple[str, str]], max_workers: int = None) -> int:
    """Convert multiple EML files to PDF in parallel. Returns number of successful conversions."""

    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2)  # Use half the CPU threads

    def convert_with_progress(pair):
        eml_path, pdf_path = pair
        success = convert_eml_to_pdf(eml_path, pdf_path)
        if success:
            print(f"   ✅ Converted: {Path(eml_path).name}")
        return success

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(convert_with_progress, eml_pdf_pairs))

    return sum(results)

def match_invoices_to_statements(invoices: List[InvoiceData],
                                transactions: List[StatementTransaction]) -> List[Dict]:
    """
    Match invoices to statement transactions based on date and amount.
    """
    matches = []
    used_transactions = set()  # Track used transactions





    for invoice in invoices:
        invoice_date_norm = normalize_date(invoice.date)

        best_match = None
        for i, transaction in enumerate(transactions):
            if i in used_transactions:  # Skip already used transactions
                continue

            trans_date_norm = normalize_date(transaction.date)

            # Check if dates and amounts match
            if (invoice_date_norm == trans_date_norm and
                abs(invoice.amount_usd - transaction.amount_usd) < 0.01):

                best_match = transaction
                used_transactions.add(i)  # Mark as used
                break
        
        match_record = {
            'invoice_number': invoice.invoice_number,
            'invoice_date': invoice.date,
            'invoice_amount_usd': invoice.amount_usd,
            'statement_type': best_match.statement_type if best_match else '',
            'statement_month': best_match.statement_file if best_match else '',
            'statement_date': best_match.date if best_match else '',
            'statement_amount_usd': best_match.amount_usd if best_match else '',
            'statement_amount_cad': best_match.amount_cad if best_match and best_match.amount_cad else '',
            'matched': bool(best_match),
            'invoice_file': Path(invoice.file_path).name,
            'statement_file': best_match.statement_file if best_match else '',
            'statement_description': best_match.description if best_match else '',
            'transaction_index': getattr(best_match, 'sequential_index', -1) if best_match else -1
        }
        
        matches.append(match_record)
    
    return matches

def relaxed_match_unmatched_invoices(matches: List[Dict], all_transactions: List[StatementTransaction]) -> List[Dict]:
    """
    Apply relaxed matching (month + amount) to unmatched invoices only.
    """
    print("\n🔄 Step 4b: Applying relaxed matching to unmatched invoices...")

    # Group transactions by month and amount for quick lookup
    month_amount_transactions = {}
    used_transactions = set()

    # First, mark all already-matched transactions as used
    for match in matches:
        if match['matched'] and 'transaction_index' in match:
            transaction_idx = match['transaction_index']
            if transaction_idx >= 0:
                used_transactions.add(transaction_idx)

    # Group available transactions by month-year and amount
    for i, transaction in enumerate(all_transactions):
        if i in used_transactions:
            continue

        # Extract month-year from transaction date
        trans_date_norm = normalize_date(transaction.date)
        if trans_date_norm:
            month_year = trans_date_norm[:7]  # "2025-04" format
            amount = transaction.amount_usd
            key = f"{month_year}_{amount}"

            if key not in month_amount_transactions:
                month_amount_transactions[key] = []
            month_amount_transactions[key].append((i, transaction))

    # Try to match unmatched invoices
    new_matches = 0

    for match in matches:
        if not match['matched']:  # Only process unmatched invoices
            invoice_date = match['invoice_date']
            invoice_amount = float(match['invoice_amount_usd'])

            # Extract month-year from invoice date
            invoice_date_norm = normalize_date(invoice_date)
            if invoice_date_norm:
                month_year = invoice_date_norm[:7]  # "2025-04" format
                key = f"{month_year}_{invoice_amount}"

                # Look for matching transactions in same month with same amount
                if key in month_amount_transactions and month_amount_transactions[key]:
                    # Pick the earliest transaction in that month
                    available_transactions = month_amount_transactions[key]
                    available_transactions.sort(key=lambda x: x[1].date)  # Sort by date

                    transaction_idx, best_transaction = available_transactions[0]

                    # Update the match
                    match['statement_type'] = best_transaction.statement_type
                    match['statement_month'] = best_transaction.statement_file
                    match['statement_date'] = best_transaction.date
                    match['statement_amount_usd'] = best_transaction.amount_usd
                    match['statement_amount_cad'] = best_transaction.amount_cad if best_transaction.amount_cad else ''
                    match['matched'] = True
                    match['statement_file'] = best_transaction.statement_file
                    match['statement_description'] = best_transaction.description
                    match['transaction_index'] = getattr(best_transaction, 'sequential_index', -1)

                    # Mark this transaction as used
                    used_transactions.add(transaction_idx)
                    month_amount_transactions[key].remove((transaction_idx, best_transaction))

                    new_matches += 1
                    print(f"   ✅ Relaxed match: {match['invoice_number']} ({invoice_date}) → {best_transaction.date}")

    print(f"   📊 Relaxed matching found {new_matches} additional matches")
    return matches

def match_orphaned_transactions(matches: List[Dict], all_transactions: List[StatementTransaction]) -> List[Dict]:
    """
    Try to match remaining orphaned transactions with orphaned invoices using very relaxed criteria.
    """
    print("\n🔄 Step 4c: Matching orphaned transactions with orphaned invoices...")

    # Find orphaned transactions (detected but not matched to any invoice)
    used_transactions = set()
    for match in matches:
        if match['matched'] and 'transaction_index' in match:
            transaction_idx = match['transaction_index']
            if transaction_idx >= 0:
                used_transactions.add(transaction_idx)

    orphaned_transactions = []
    for i, transaction in enumerate(all_transactions):
        if i not in used_transactions:
            orphaned_transactions.append((i, transaction))

    # Find orphaned invoices (unmatched)
    orphaned_invoices = [match for match in matches if not match['matched']]

    print(f"   🔍 Found {len(orphaned_transactions)} orphaned transactions")
    print(f"   🔍 Found {len(orphaned_invoices)} orphaned invoices")

    # Try to match orphaned transactions with orphaned invoices
    # Use very relaxed criteria: similar amounts (within $5) regardless of date
    new_matches = 0

    for match in orphaned_invoices:
        if match['matched']:  # Skip if already matched in previous iterations
            continue

        invoice_amount = float(match['invoice_amount_usd'])
        best_transaction = None
        best_transaction_idx = None
        best_score = float('inf')

        for transaction_idx, transaction in orphaned_transactions:
            # Calculate similarity score (amount difference)
            amount_diff = abs(transaction.amount_usd - invoice_amount)

            # Only consider if amount is within $5
            if amount_diff <= 5.0:
                # Prefer closer amounts
                score = amount_diff

                if score < best_score:
                    best_score = score
                    best_transaction = transaction
                    best_transaction_idx = transaction_idx

        # If we found a reasonable match, use it
        if best_transaction and best_score <= 6.0:
            # Update the match
            match['statement_type'] = best_transaction.statement_type
            match['statement_month'] = best_transaction.statement_file
            match['statement_date'] = best_transaction.date
            match['statement_amount_usd'] = best_transaction.amount_usd
            match['statement_amount_cad'] = best_transaction.amount_cad if best_transaction.amount_cad else ''
            match['matched'] = True
            match['statement_file'] = best_transaction.statement_file
            match['statement_description'] = best_transaction.description
            match['transaction_index'] = getattr(best_transaction, 'sequential_index', -1)

            # Remove this transaction from orphaned list
            orphaned_transactions = [(idx, t) for idx, t in orphaned_transactions if idx != best_transaction_idx]

            new_matches += 1
            amount_diff = abs(best_transaction.amount_usd - invoice_amount)
            print(f"   ✅ Orphan match: {match['invoice_number']} (${invoice_amount}) → {best_transaction.date} (${best_transaction.amount_usd}) [diff: ${amount_diff:.2f}]")

    print(f"   📊 Orphaned matching found {new_matches} additional matches")
    return matches

def highlight_statement_matches(statement_pdf_path: str, matched_transactions: List[StatementTransaction],
                               output_dir: str) -> str:
    """
    Create a highlighted copy of the statement PDF with matched transactions highlighted.

    Returns the path to the highlighted PDF.
    """
    if not PDF_HIGHLIGHT_AVAILABLE:
        print(f"   ⚠️  Skipping highlighting for {Path(statement_pdf_path).name} - PDF libraries not available")
        return ""

    try:
        import fitz  # PyMuPDF

        # Open the PDF
        doc = fitz.open(statement_pdf_path)

        # Get the statement filename for output
        statement_name = Path(statement_pdf_path).stem
        output_path = Path(output_dir) / f"{statement_name}_HIGHLIGHTED.pdf"

        # Simple highlighting strategy: one pattern per transaction
        statement_name = Path(statement_pdf_path).name
        print(f"   🎨 Highlighting {len(matched_transactions)} transactions in {statement_name}")

        highlighted_count = 0

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]

            # For each transaction, find ONE specific pattern to highlight
            for t in matched_transactions:
                pattern = None

                # For credit statements, use ref number
                if 'credit' in statement_name.lower():
                    import re
                    ref_match = re.match(r'(\d{3})\s+', t.description)
                    if ref_match:
                        pattern = ref_match.group(1)

                # For bank statements, use a more specific pattern
                else:
                    import re
                    # Extract amount from description like "Opos25.00AugmentCode+1408357014CAUS"
                    amount_match = re.search(r'Opos(\d+\.\d{2})', t.description)
                    company_match = re.search(r'(AugmentCode|Openrouter)', t.description, re.IGNORECASE)

                    if amount_match and company_match:
                        amount = amount_match.group(1)
                        company = company_match.group(1)

                        # Try to search for a more specific pattern that combines amount and company
                        # This should be more unique than just the amount
                        if 'augment' in company.lower():
                            pattern = f"AugmentCode"  # Search for the company name specifically
                        else:
                            pattern = f"Openrouter"

                # Search and highlight this pattern (only once per transaction)
                if pattern:
                    text_instances = page.search_for(pattern)
                    if text_instances:
                        # Only highlight the first instance to avoid duplicates
                        inst = text_instances[0]
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                        highlight.update()
                        highlighted_count += 1



        # Save the highlighted PDF
        doc.save(str(output_path))
        doc.close()

        print(f"   ✅ Highlighted PDF saved: {output_path.name} ({highlighted_count} highlights)")
        return str(output_path)

    except Exception as e:
        print(f"   ❌ Error highlighting {Path(statement_pdf_path).name}: {e}")
        return ""

def create_highlighted_statements(matches: List[Dict], temp_extract_dir: str, output_dir: str):
    """
    Create highlighted copies of all statement PDFs showing matched transactions.
    Uses CSV data to precisely highlight only matched transactions.
    """
    if not PDF_HIGHLIGHT_AVAILABLE:
        print("   ⚠️  PDF highlighting not available - skipping highlighted statement creation")
        return

    print("\n🎨 Step 6: Creating highlighted statement PDFs...")

    # Create output directory for highlighted PDFs
    highlighted_dir = Path(output_dir) / "highlighted_statements"
    highlighted_dir.mkdir(exist_ok=True)

    # Read the CSV to get exact matches
    csv_path = Path(output_dir) / "financial_reconciliation.csv"
    if not csv_path.exists():
        print("   ⚠️  CSV file not found - cannot create highlighted statements")
        return

    # Group CSV matches by statement file
    import csv
    matches_by_statement = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['matched'] == 'True' and row['invoice_file']:
                statement_file = row['statement_file']
                if statement_file not in matches_by_statement:
                    matches_by_statement[statement_file] = []
                matches_by_statement[statement_file].append(row)

    # Create highlighted PDFs for each statement
    highlighted_files = []
    for statement_file, csv_matches in matches_by_statement.items():
        statement_path = Path(temp_extract_dir) / statement_file

        if statement_path.exists():
            highlighted_path = highlight_statement_from_csv(
                str(statement_path),
                csv_matches,
                str(highlighted_dir)
            )
            if highlighted_path:
                highlighted_files.append(highlighted_path)
        else:
            print(f"   ⚠️  Statement file not found: {statement_file}")

    print(f"   📄 Created {len(highlighted_files)} highlighted statement PDFs")
    print(f"   📁 Highlighted PDFs saved in: {highlighted_dir}")

def add_cad_highlights_for_bank_statement(doc):
    """Add CAD amount highlights for bank statements by finding amounts 2 lines above existing USD highlights."""
    import re
    cad_highlights_added = 0
    used_cad_instances = set()

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get all existing highlights (these are USD transactions)
        existing_highlights = []
        for annot in page.annots():
            if annot.type[1] == 'Highlight':
                existing_highlights.append(annot.rect)

        if not existing_highlights:
            continue

        # Get page text lines
        page_text = page.get_text()
        lines = page_text.split('\n')

        print(f"   📄 Page {page_num + 1}: Found {len(existing_highlights)} existing USD highlights")

        # For each existing USD highlight, find the CAD amount above it
        for highlight_rect in existing_highlights:
            # Find which line this highlight corresponds to
            usd_line_index = None

            # Search through lines to find the one that matches this highlight position
            for i, line in enumerate(lines):
                if line.strip() and 'Opos' in line:
                    # Check if this line is at roughly the same Y position as our highlight
                    line_instances = page.search_for(line.strip())
                    for line_rect in line_instances:
                        if abs(line_rect.y0 - highlight_rect.y0) < 15:  # Allow some tolerance
                            usd_line_index = i
                            break
                    if usd_line_index is not None:
                        break

            if usd_line_index is not None and usd_line_index >= 2:
                # Get the line 2 positions above
                cad_line_index = usd_line_index - 2
                potential_cad_line = lines[cad_line_index].strip()

                # Check if this line contains a CAD amount (decimal number)
                cad_match = re.match(r'^(\d+\.\d{2})$', potential_cad_line)

                if cad_match:
                    cad_amount = cad_match.group(1)

                    # Find and highlight this CAD amount
                    cad_instances = page.search_for(cad_amount)
                    for cad_idx, cad_inst in enumerate(cad_instances):
                        # Check if this instance is above our USD highlight
                        if cad_inst.y1 < highlight_rect.y0:
                            cad_key = (page_num, cad_amount, cad_idx)
                            if cad_key not in used_cad_instances:
                                cad_highlight = page.add_highlight_annot(cad_inst)
                                cad_highlight.set_colors(stroke=[1, 0, 0])  # Red
                                cad_highlight.update()
                                used_cad_instances.add(cad_key)
                                cad_highlights_added += 1
                                print(f"   💰 Added CAD highlight: {cad_amount}")
                                break

    return cad_highlights_added


def highlight_statement_from_csv(statement_pdf_path: str, csv_matches: List[Dict], output_dir: str) -> str:
    """
    Create a highlighted copy of the statement PDF using exact CSV match data.
    Iterates through each CSV row and finds/highlights the corresponding text in the PDF.
    """
    if not PDF_HIGHLIGHT_AVAILABLE:
        print(f"   ⚠️  Skipping highlighting for {Path(statement_pdf_path).name} - PDF libraries not available")
        return ""

    try:
        import fitz  # PyMuPDF

        # Open the PDF
        doc = fitz.open(statement_pdf_path)

        # Get the statement filename for output
        statement_name = Path(statement_pdf_path).stem
        output_path = Path(output_dir) / f"{statement_name}_HIGHLIGHTED.pdf"

        print(f"   🎨 Highlighting {len(csv_matches)} transactions in {statement_name}.pdf")

        # Sort CSV matches by transaction_index (lowest to highest) for consistent processing
        sorted_csv_matches = sorted(csv_matches, key=lambda x: int(x['transaction_index']) if x['transaction_index'].isdigit() else 999999)

        # Debug: count how many instances of each pattern exist
        debug_patterns = {}
        for match in sorted_csv_matches:
            if 'credit' in statement_name.lower() and 'augment' in match['statement_description'].lower():
                pattern = "AUGMENT CODE"
            elif 'credit' in statement_name.lower() and 'openrouter' in match['statement_description'].lower():
                pattern = "OPENROUTER"
            else:
                pattern = "bank_pattern"

            debug_patterns[pattern] = debug_patterns.get(pattern, 0) + 1

        print(f"   🎨 Expected patterns: {debug_patterns}")

        highlighted_count = 0
        used_instances = set()  # Track (page_num, pattern, instance_index) that have been used

        # Process each CSV match once across all pages (fixed loop structure)
        highlighted_matches = set()  # Track which CSV matches have been highlighted

        for match_index, match in enumerate(sorted_csv_matches):
            if match_index in highlighted_matches:
                continue  # Skip already highlighted matches

            search_pattern = None
            import re

            # Extract search pattern based on statement type
            if 'credit' in statement_name.lower():
                # For credit statements: we need to highlight both company name AND amount
                # We'll search for the amount pattern first, then also highlight the company name
                search_patterns = []  # List of patterns to search for this transaction

                usd_amount = match['statement_amount_usd']
                # Format amount to match PDF format (e.g., "25.0" -> "25.00")
                if '.' in usd_amount and len(usd_amount.split('.')[1]) == 1:
                    usd_amount = usd_amount + '0'  # 25.0 -> 25.00

                if 'augment' in match['statement_description'].lower():
                    search_patterns = [f"AMT {usd_amount} USD", "AUGMENT CODE"]
                elif 'openrouter' in match['statement_description'].lower():
                    search_patterns = [f"AMT {usd_amount} USD", "OPENROUTER"]

                # For now, use the amount pattern as primary (we'll enhance this)
                search_pattern = search_patterns[0] if search_patterns else None

            else:
                # For bank statements: "Opos25.00AugmentCode+1408357014CAUS"
                # PDF: "Opos 25.00 Augment Code +1408357014CAUS"
                amount_match = re.search(r'Opos(\d+\.\d{2})', match['statement_description'])
                if amount_match:
                    amount_str = amount_match.group(1)
                    if 'augment' in match['statement_description'].lower():
                        search_pattern = f"Opos {amount_str} Augment Code"
                    elif 'openrouter' in match['statement_description'].lower():
                        search_pattern = f"Opos {amount_str} Openrouter"

            # Search for this pattern across all pages
            if search_pattern:
                found_highlight_for_this_csv_row = False

                for page_num in range(len(doc)):
                    if found_highlight_for_this_csv_row:
                        break  # Already found a highlight for this CSV row

                    page = doc[page_num]
                    text_instances = page.search_for(search_pattern)

                    if text_instances:
                        print(f"   🔍 CSV row {match_index + 1}: Found {len(text_instances)} instances of '{search_pattern}' on page {page_num + 1}")

                    # Find the next available instance that hasn't been used yet
                    for inst_index, inst in enumerate(text_instances):
                        # Create unique key for this specific instance
                        instance_key = (page_num, search_pattern, inst_index)

                        if instance_key not in used_instances:
                            # Highlight this instance
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                            highlight.update()
                            highlighted_count += 1
                            used_instances.add(instance_key)

                            # For credit statements, also try to highlight the company name and CAD amount
                            if 'credit' in statement_name.lower():
                                company_pattern = None
                                if 'augment' in match['statement_description'].lower():
                                    company_pattern = "AUGMENT CODE"
                                elif 'openrouter' in match['statement_description'].lower():
                                    company_pattern = "OPENROUTER"

                                if company_pattern:
                                    # Search for company name on this page
                                    company_instances = page.search_for(company_pattern)
                                    for comp_idx, comp_inst in enumerate(company_instances):
                                        comp_key = (page_num, company_pattern, comp_idx)
                                        if comp_key not in used_instances:
                                            # Highlight company name too
                                            comp_highlight = page.add_highlight_annot(comp_inst)
                                            comp_highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                                            comp_highlight.update()
                                            highlighted_count += 1
                                            used_instances.add(comp_key)
                                            break  # Only highlight one company instance per transaction

                                # Also highlight the CAD amount from CSV description (CREDIT CARD APPROACH)
                                import re
                                cad_match = None
                                if 'augment' in match['statement_description'].lower():
                                    cad_match = re.search(r'AUGMENT CODE\s+(\d+\.\d{2})', match['statement_description'])
                                elif 'openrouter' in match['statement_description'].lower():
                                    cad_match = re.search(r'OPENROUTER\s+(\d+\.\d{2})', match['statement_description'])

                                if cad_match:
                                    cad_amount = cad_match.group(1)
                                    # Search for this CAD amount on the current page
                                    cad_instances = page.search_for(cad_amount)
                                    for cad_idx, cad_inst in enumerate(cad_instances):
                                        cad_key = (page_num, f"CAD_{cad_amount}", cad_idx)
                                        if cad_key not in used_instances:
                                            # Highlight CAD amount
                                            cad_highlight = page.add_highlight_annot(cad_inst)
                                            cad_highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                                            cad_highlight.update()
                                            highlighted_count += 1
                                            used_instances.add(cad_key)
                                            print(f"   💰 Also highlighted credit CAD amount: {cad_amount}")
                                            break  # Only highlight first unused CAD instance

                            highlighted_matches.add(match_index)
                            found_highlight_for_this_csv_row = True
                            print(f"   ✅ CSV row {match_index + 1}: Highlighted instance {inst_index + 1}/{len(text_instances)} of '{search_pattern}' on page {page_num + 1}")

                            # For bank statements: Add CAD highlighting immediately
                            if 'bank' in statement_name.lower():
                                # Simple approach: find the CAD amount on the same row as the USD transaction
                                # Look for decimal amounts that appear before the USD transaction on the same line

                                # Get the text of the entire row containing our USD transaction
                                row_text = page.get_textbox(fitz.Rect(0, inst.y0 - 5, page.rect.width, inst.y1 + 5))

                                # Find all decimal amounts in this row
                                import re
                                amounts = re.findall(r'\b(\d+\.\d{2})\b', row_text)

                                # The CAD amount should be the first decimal amount in the row (before the USD transaction)
                                if amounts:
                                    cad_amount = amounts[0]  # First amount is usually the CAD amount

                                    # Highlight this CAD amount
                                    cad_instances = page.search_for(cad_amount)
                                    for cad_idx, cad_inst in enumerate(cad_instances):
                                        # Find the instance that's on the same row as our USD transaction
                                        if abs(cad_inst.y0 - inst.y0) < 10:  # Same row
                                            cad_key = (page_num, f"CAD_{cad_amount}", cad_idx)
                                            if cad_key not in used_instances:
                                                cad_highlight = page.add_highlight_annot(cad_inst)
                                                cad_highlight.set_colors(stroke=[1, 0, 0])  # Red
                                                cad_highlight.update()
                                                highlighted_count += 1
                                                used_instances.add(cad_key)
                                                print(f"   💰 Added CAD highlight: {cad_amount}")
                                                break

                            break  # Found one for this CSV match, move to next CSV row
                        else:
                            print(f"   ⏭️  CSV row {match_index + 1}: Skipping instance {inst_index + 1}/{len(text_instances)} of '{search_pattern}' on page {page_num + 1} (already used)")

                # Debug: if we couldn't find an unhighlighted instance
                if not found_highlight_for_this_csv_row:
                    print(f"   ⚠️  Could not highlight CSV row {match_index + 1} with pattern '{search_pattern}'")

        # CAD highlighting is now done inline with USD highlighting

        # Save the highlighted PDF
        doc.save(str(output_path))
        doc.close()

        print(f"   ✅ Highlighted PDF saved: {output_path.name} ({highlighted_count} highlights)")
        return str(output_path)

    except Exception as e:
        print(f"   ❌ Error highlighting {Path(statement_pdf_path).name}: {e}")
        return ""

def rects_overlap(rect1, rect2) -> bool:
    """Check if two rectangles are essentially the same (for avoiding duplicate highlights)."""
    # Instead of checking for any overlap, check if they're essentially the same rectangle
    # Allow small tolerance for floating point differences
    tolerance = 2.0

    same_position = (abs(rect1.x0 - rect2.x0) < tolerance and
                    abs(rect1.y0 - rect2.y0) < tolerance and
                    abs(rect1.x1 - rect2.x1) < tolerance and
                    abs(rect1.y1 - rect2.y1) < tolerance)

    return same_position

def filter_matches_with_invoice_files(matches: List[Dict]) -> List[Dict]:
    """
    Filter matches to only include those with invoice files (same logic as CSV writing).
    """
    return [match for match in matches if match['matched'] and match['invoice_file']]

def renumber_transaction_indices_by_pdf_order(matches: List[Dict], temp_extract_dir: str) -> List[Dict]:
    """
    Renumber transaction_index based on actual PDF appearance order (1, 2, 3...).
    """
    def normalize_amount(amount_str):
        """Normalize amount strings for comparison (10.9 -> 10.90)"""
        try:
            return f"{float(amount_str):.2f}"
        except:
            return amount_str

    # Group matches by statement file
    by_statement = {}
    for match in matches:
        if match['matched'] and match['statement_file']:
            statement_file = match['statement_file']
            if statement_file not in by_statement:
                by_statement[statement_file] = []
            by_statement[statement_file].append(match)

    updated_matches = []

    for match in matches:
        updated_match = match.copy()

        if match['matched'] and match['statement_file']:
            statement_file = match['statement_file']
            statement_matches = by_statement[statement_file]

            # Scan PDF to get actual transaction order
            pdf_path = Path(temp_extract_dir) / statement_file
            if pdf_path.exists():
                try:
                    import fitz
                    doc = fitz.open(str(pdf_path))
                    pdf_transactions = []

                    # Scan all pages for transactions in order
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text()
                        lines = text.split('\n')

                        for line in lines:
                            if 'credit' in statement_file.lower():
                                # Credit: look for AMT patterns
                                if line.startswith('AMT ') and 'USD' in line:
                                    import re
                                    amount_match = re.search(r'AMT (\d+\.\d{2}) USD', line)
                                    if amount_match:
                                        amount = amount_match.group(1)
                                        pdf_transactions.append(normalize_amount(amount))
                            else:
                                # Bank: look for Opos patterns
                                if 'Opos ' in line and ('Openrouter' in line or 'Augment' in line):
                                    import re
                                    amount_match = re.search(r'Opos (\d+\.\d{2})', line)
                                    if amount_match:
                                        amount = amount_match.group(1)
                                        pdf_transactions.append(normalize_amount(amount))

                    doc.close()

                    # Match CSV entries to PDF order and assign indices
                    csv_amount = normalize_amount(match['statement_amount_usd'])

                    # Find first matching amount in PDF order
                    for pdf_pos, pdf_amount in enumerate(pdf_transactions):
                        if pdf_amount == csv_amount:
                            updated_match['transaction_index'] = pdf_pos + 1  # 1-based
                            break
                    else:
                        # Fallback: use original index if no match found
                        updated_match['transaction_index'] = match['transaction_index']

                except Exception as e:
                    print(f"   ⚠️  Error scanning PDF {statement_file}: {e}")
                    # Fallback: keep original index
                    updated_match['transaction_index'] = match['transaction_index']
            else:
                # PDF not found, keep original index
                updated_match['transaction_index'] = match['transaction_index']

        updated_matches.append(updated_match)

    return updated_matches

def create_monthly_folders(matches: List[Dict], base_dir: str, temp_extract_dir: str, output_dir: str):
    """
    Create monthly folders organized by statement files, with all invoices paid in each statement.
    """
    print("\n📁 Step 7: Creating statement-based monthly folders...")

    # Create monthly folders directory
    monthly_dir = Path(output_dir) / "monthly_folders"
    monthly_dir.mkdir(exist_ok=True)

    # Group matches by statement file
    statement_data = {}
    unmatched_invoices = []

    for match in matches:
        if match['matched'] and match['statement_file']:
            statement_file = match['statement_file']
            if statement_file not in statement_data:
                statement_data[statement_file] = []
            statement_data[statement_file].append(match)
        else:
            # Keep track of unmatched invoices
            unmatched_invoices.append(match)

    # Extract month/year from statement filenames and group
    monthly_data = {}

    for statement_file, statement_matches in statement_data.items():
        # Extract month from statement filename
        # Examples: "bank_May 2025 e-statement.pdf", "credit_July 2025 e-statement.pdf"
        month_year = None

        # Look for month patterns in filename (both full names and abbreviations)
        import re
        month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s_]+(\d{4})', statement_file, re.IGNORECASE)
        if month_match:
            month_year = f"{month_match.group(1)} {month_match.group(2)}"

        if month_year:
            if month_year not in monthly_data:
                monthly_data[month_year] = {
                    'matches': [],
                    'statement_files': set()
                }
            monthly_data[month_year]['matches'].extend(statement_matches)
            monthly_data[month_year]['statement_files'].add(statement_file)

    # Add unmatched invoices to a separate folder if there are any
    if unmatched_invoices:
        monthly_data['Unmatched'] = {
            'matches': unmatched_invoices,
            'statement_files': set()
        }

    # Create folders and copy files for each statement month
    for month_year, month_data_dict in monthly_data.items():
        month_matches = month_data_dict['matches']
        statement_files = month_data_dict['statement_files']

        print(f"   📅 Processing {month_year}...")

        # Create month folder with subfolders
        month_folder = monthly_dir / month_year.replace(' ', '_')
        month_folder.mkdir(exist_ok=True)

        # Create subfolders
        augment_folder = month_folder / "augment"
        openrouter_folder = month_folder / "openrouter"
        combined_folder = month_folder / "combined"

        augment_folder.mkdir(exist_ok=True)
        openrouter_folder.mkdir(exist_ok=True)
        combined_folder.mkdir(exist_ok=True)

        # Separate matches by type
        augment_matches = []
        openrouter_matches = []

        for match in month_matches:
            if match['invoice_number'].startswith('OR-'):
                openrouter_matches.append(match)
            else:
                augment_matches.append(match)

        # Copy invoice files to appropriate subfolders and combined
        augment_files_copied = 0
        openrouter_files_copied = 0

        import shutil

        # Process Augment invoices
        for match in augment_matches:
            invoice_file = match['invoice_file']
            source_file = Path(base_dir).parent / "extracted_attachments_renamed" / invoice_file
            if not source_file.exists():
                # Try finding in subdirectories
                for pdf_file in Path(base_dir).parent.rglob(invoice_file):
                    source_file = pdf_file
                    break

            if source_file and source_file.exists():
                # Copy to augment subfolder
                dest_file = augment_folder / invoice_file
                if not dest_file.exists():
                    shutil.copy2(source_file, dest_file)
                    augment_files_copied += 1

                # Copy to combined folder
                combined_dest = combined_folder / invoice_file
                if not combined_dest.exists():
                    shutil.copy2(source_file, combined_dest)

        # Collect EML files for parallel conversion
        eml_conversions = []
        pdf_copies = []

        for match in openrouter_matches:
            invoice_file = match['invoice_file']
            source_file = Path(base_dir) / invoice_file

            if source_file and source_file.exists():
                pdf_filename = invoice_file.replace('.eml', '.pdf')

                if invoice_file.endswith('.eml'):
                    # Add to parallel conversion list
                    dest_file = openrouter_folder / pdf_filename
                    combined_dest = combined_folder / pdf_filename

                    if not dest_file.exists():
                        eml_conversions.append((str(source_file), str(dest_file)))
                    if not combined_dest.exists():
                        eml_conversions.append((str(source_file), str(combined_dest)))
                else:
                    # Add to regular copy list
                    pdf_copies.append((source_file, openrouter_folder / pdf_filename, combined_folder / pdf_filename))

        # Process EML conversions in parallel
        if eml_conversions:
            workers = max(1, os.cpu_count() // 2)
            print(f"   🔄 Converting {len(eml_conversions)} EML files to PDF using {workers} workers...")
            successful_conversions = convert_emls_to_pdfs_parallel(eml_conversions)
            openrouter_files_copied += successful_conversions // 2  # Each file goes to 2 locations

        # Process regular PDF copies
        for source_file, dest_file, combined_dest in pdf_copies:
            if not dest_file.exists():
                shutil.copy2(source_file, dest_file)
                openrouter_files_copied += 1
            if not combined_dest.exists():
                shutil.copy2(source_file, combined_dest)

        # Copy statement files to all subfolders (prefer highlighted versions)
        statement_files_copied = 0
        highlighted_dir = Path(output_dir) / "highlighted_statements"

        for statement_file in statement_files:
            # Try to copy highlighted version first
            statement_name = Path(statement_file).stem
            highlighted_file = highlighted_dir / f"{statement_name}_HIGHLIGHTED.pdf"

            if highlighted_file.exists():
                # Copy highlighted version to all subfolders (always overwrite to get latest highlights)
                for subfolder in [augment_folder, openrouter_folder, combined_folder]:
                    dest_file = subfolder / f"{statement_name}_HIGHLIGHTED.pdf"
                    shutil.copy2(highlighted_file, dest_file)  # Always copy to get latest highlights
                statement_files_copied += 1
            else:
                # Fallback to original statement
                source_file = Path(temp_extract_dir) / statement_file
                if source_file.exists():
                    for subfolder in [augment_folder, openrouter_folder, combined_folder]:
                        dest_file = subfolder / statement_file
                        if not dest_file.exists():
                            shutil.copy2(source_file, dest_file)
                    statement_files_copied += 1

        # Create CSV files for each subfolder
        # Combined CSV
        combined_csv = combined_folder / f"{month_year.replace(' ', '_')}_combined_reconciliation.csv"
        with open(combined_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'invoice_number', 'invoice_date', 'invoice_amount_usd',
                'statement_type', 'statement_month', 'statement_date', 'statement_amount_usd', 'statement_amount_cad',
                'matched', 'invoice_file', 'statement_file', 'statement_description', 'transaction_index'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Update .eml filenames to .pdf and renumber transaction indices
            updated_matches = []
            for match in month_matches:
                updated_match = match.copy()
                if updated_match['invoice_file'].endswith('.eml'):
                    updated_match['invoice_file'] = updated_match['invoice_file'].replace('.eml', '.pdf')
                updated_matches.append(updated_match)

            # Sort by transaction_index for consistent CSV order (don't change the indices, just sort)
            sorted_matches = sorted(updated_matches, key=lambda x: int(x['transaction_index']) if str(x['transaction_index']).isdigit() else 999999)
            writer.writerows(sorted_matches)

        # Augment CSV
        if augment_matches:
            augment_csv = augment_folder / f"{month_year.replace(' ', '_')}_augment_reconciliation.csv"
            with open(augment_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'invoice_number', 'invoice_date', 'invoice_amount_usd',
                    'statement_type', 'statement_month', 'statement_date', 'statement_amount_usd', 'statement_amount_cad',
                    'matched', 'invoice_file', 'statement_file', 'statement_description', 'transaction_index'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Update .eml filenames to .pdf and renumber transaction indices
                updated_augment = []
                for match in augment_matches:
                    updated_match = match.copy()
                    if updated_match['invoice_file'].endswith('.eml'):
                        updated_match['invoice_file'] = updated_match['invoice_file'].replace('.eml', '.pdf')
                    updated_augment.append(updated_match)

                # Sort by transaction_index for consistent CSV order (don't change the indices, just sort)
                sorted_augment = sorted(updated_augment, key=lambda x: int(x['transaction_index']) if str(x['transaction_index']).isdigit() else 999999)
                writer.writerows(sorted_augment)

        # OpenRouter CSV
        if openrouter_matches:
            openrouter_csv = openrouter_folder / f"{month_year.replace(' ', '_')}_openrouter_reconciliation.csv"
            with open(openrouter_csv, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'invoice_number', 'invoice_date', 'invoice_amount_usd',
                    'statement_type', 'statement_month', 'statement_date', 'statement_amount_usd', 'statement_amount_cad',
                    'matched', 'invoice_file', 'statement_file', 'statement_description', 'transaction_index'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Update .eml filenames to .pdf and renumber transaction indices
                updated_openrouter = []
                for match in openrouter_matches:
                    updated_match = match.copy()
                    if updated_match['invoice_file'].endswith('.eml'):
                        updated_match['invoice_file'] = updated_match['invoice_file'].replace('.eml', '.pdf')
                    updated_openrouter.append(updated_match)

                # Sort by transaction_index for consistent CSV order (don't change the indices, just sort)
                sorted_openrouter = sorted(updated_openrouter, key=lambda x: int(x['transaction_index']) if str(x['transaction_index']).isdigit() else 999999)
                writer.writerows(sorted_openrouter)

        matched_count = sum(1 for m in month_matches if m['matched'])
        augment_matched = sum(1 for m in augment_matches if m['matched'])
        openrouter_matched = sum(1 for m in openrouter_matches if m['matched'])

        print(f"     ✅ {month_year}: {len(month_matches)} total invoices, {matched_count} matched")
        print(f"        📧 Augment: {len(augment_matches)} invoices, {augment_matched} matched")
        print(f"        📧 OpenRouter: {len(openrouter_matches)} invoices, {openrouter_matched} matched")
        print(f"        📄 {augment_files_copied + openrouter_files_copied} invoice files, {statement_files_copied} statement files")

    print(f"   📁 Monthly folders created in: {monthly_dir}")

def main():
    print("💰 Financial Reconciliation Script")
    print("=" * 50)
    
    # Define paths
    base_dir = "/home/ts/Downloads/coding_llm_provider_invoices"
    augment_invoices_dir = "/home/ts/Downloads/extracted_attachments_renamed"
    temp_extract_dir = "/home/ts/Downloads/temp_statements"
    output_csv = "/home/ts/Downloads/financial_reconciliation.csv"
    
    # Create temp directory
    Path(temp_extract_dir).mkdir(exist_ok=True)

    print("📦 Step 1: Extracting ZIP files...")
    
    # Extract ZIP files with prefixes
    bank_files = extract_zip_with_prefix(
        f"{base_dir}/bank_statements.zip", 
        temp_extract_dir, 
        "bank"
    )
    
    credit_files = extract_zip_with_prefix(
        f"{base_dir}/credit_statements.zip", 
        temp_extract_dir, 
        "credit"
    )
    
    print(f"   ✅ Extracted {len(bank_files)} bank statements")
    print(f"   ✅ Extracted {len(credit_files)} credit statements")
    
    print("\n📄 Step 2: Processing invoices...")

    # Copy and rename Augment PDFs from extracted_attachments
    extracted_attachments_dir = "/home/ts/Downloads/extracted_attachments"
    if Path(extracted_attachments_dir).exists():
        Path(augment_invoices_dir).mkdir(parents=True, exist_ok=True)

        for folder in Path(extracted_attachments_dir).iterdir():
            if folder.is_dir():
                # Find PDF files in this folder
                for pdf_file in folder.glob("*.pdf"):
                    # Create filename from folder name
                    # "Payment received for Augment Code invoice (#EQCIET-00045)"
                    # becomes "Payment_received_for_Augment_Code_invoice-EQCIET-00045.pdf"
                    folder_name = folder.name

                    # Clean up the folder name for filename
                    clean_name = folder_name.replace(" ", "_")
                    clean_name = clean_name.replace("(#", "-")
                    clean_name = clean_name.replace(")", "")
                    clean_name = clean_name.replace("#", "")

                    dest_filename = f"{clean_name}.pdf"
                    dest_file = Path(augment_invoices_dir) / dest_filename

                    shutil.copy2(pdf_file, dest_file)
                    print(f"   ✅ Renamed: {folder.name} → {dest_filename}")
    else:
        print(f"   ⚠️  Extracted attachments directory not found: {extracted_attachments_dir}")

    # Process invoices
    invoices = []

    # Process Augment invoices (PDF attachments)
    if Path(augment_invoices_dir).exists():
        print("   📧 Processing Augment invoice PDFs...")
        invoice_files = list(Path(augment_invoices_dir).rglob("*.pdf"))

        for pdf_path in invoice_files:
            invoice_data = parse_invoice_pdf(str(pdf_path))
            if invoice_data:
                invoices.append(invoice_data)
                print(f"   ✅ Augment: {invoice_data.invoice_number} - {invoice_data.date} - ${invoice_data.amount_usd}")

    # Process OpenRouter invoices (.eml files)
    print("   📧 Processing OpenRouter invoice emails...")
    openrouter_files = list(Path(base_dir).glob("Your OpenRouter*.eml"))

    for eml_path in openrouter_files:
        invoice_data = parse_openrouter_eml(str(eml_path))
        if invoice_data:
            invoices.append(invoice_data)
            print(f"   ✅ OpenRouter: {invoice_data.invoice_number} - {invoice_data.date} - ${invoice_data.amount_usd}")

    print(f"   📊 Total invoices processed: {len(invoices)} (Augment + OpenRouter)")
    
    print("\n🏦 Step 3: Processing statement PDFs...")
    
    # Process statements
    all_transactions = []
    
    for pdf_path in bank_files:
        transactions = parse_statement_pdf(pdf_path, "bank")
        all_transactions.extend(transactions)
        print(f"   🏦 Bank: {Path(pdf_path).name} - {len(transactions)} Augment transactions")
    
    for pdf_path in credit_files:
        transactions = parse_statement_pdf(pdf_path, "credit")
        all_transactions.extend(transactions)
        print(f"   💳 Credit: {Path(pdf_path).name} - {len(transactions)} Augment transactions")
    
    print(f"   📊 Total Augment transactions found: {len(all_transactions)}")

    # Fix: Reassign sequential_index globally to ensure unique indices
    for i, transaction in enumerate(all_transactions):
        transaction.sequential_index = i

    print("\n🔗 Step 4: Matching invoices to statements...")
    
    # Match invoices to statements
    matches = match_invoices_to_statements(invoices, all_transactions)

    initial_matched_count = sum(1 for m in matches if m['matched'])
    print(f"   ✅ Initial matches: {initial_matched_count}/{len(invoices)} invoices")

    # Apply relaxed matching to unmatched invoices
    matches = relaxed_match_unmatched_invoices(matches, all_transactions)

    relaxed_matched_count = sum(1 for m in matches if m['matched'])
    print(f"   ✅ After relaxed matching: {relaxed_matched_count}/{len(invoices)} invoices (+{relaxed_matched_count - initial_matched_count} from relaxed matching)")

    # Try to match orphaned transactions with orphaned invoices
    matches = match_orphaned_transactions(matches, all_transactions)

    final_matched_count = sum(1 for m in matches if m['matched'])
    print(f"   ✅ Final matches: {final_matched_count}/{len(invoices)} invoices (+{final_matched_count - relaxed_matched_count} from orphaned matching)")

    print("\n📊 Step 5: Generating CSV reports...")

    # Write reconciliation CSV report
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'invoice_number', 'invoice_date', 'invoice_amount_usd',
            'statement_type', 'statement_month', 'statement_date', 'statement_amount_usd', 'statement_amount_cad',
            'matched', 'invoice_file', 'statement_file', 'statement_description', 'transaction_index'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by transaction_index for consistent CSV order (don't change the indices, just sort)
        sorted_matches = sorted(matches, key=lambda x: int(x['transaction_index']) if str(x['transaction_index']).isdigit() else 999999)
        writer.writerows(sorted_matches)

    print(f"   ✅ Reconciliation CSV saved: {output_csv}")

    # Write all Augment transactions CSV
    transactions_csv = "/home/ts/Downloads/all_augment_transactions.csv"
    with open(transactions_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'date', 'amount_usd', 'statement_type', 'statement_file', 'description'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for transaction in all_transactions:
            writer.writerow({
                'date': transaction.date,
                'amount_usd': transaction.amount_usd,
                'statement_type': transaction.statement_type,
                'statement_file': transaction.statement_file,
                'description': transaction.description
            })

    print(f"   ✅ All transactions CSV saved: {transactions_csv}")

    print(f"\n🎨 Step 6: Creating highlighted statement PDFs (after all matching)...")
    # Create highlighted statement PDFs using the same filtered data as CSV (no renumbering)
    filtered_matches = filter_matches_with_invoice_files(matches)

    create_highlighted_statements(filtered_matches, temp_extract_dir, os.path.dirname(output_csv))

    # Create monthly folders AFTER highlighting so highlighted PDFs get copied
    create_monthly_folders(matches, base_dir, temp_extract_dir, os.path.dirname(output_csv))

    # Summary
    print(f"\n🎉 Reconciliation Complete!")
    print(f"   📧 Invoices processed: {len(invoices)}")
    print(f"   🏦 Statement transactions: {len(all_transactions)}")
    print(f"   ✅ Successful matches: {final_matched_count}")
    print(f"   ❌ Unmatched invoices: {len(invoices) - final_matched_count}")
    print(f"   📄 Report: {output_csv}")
    print(f"   🎨 Highlighted statements: /home/ts/Downloads/highlighted_statements/")

if __name__ == "__main__":
    main()

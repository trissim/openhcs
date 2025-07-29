#!/usr/bin/env python3
"""
Inspect OpenRouter .eml files to understand their structure.
"""

import email
import re
from pathlib import Path

def inspect_openrouter_eml(eml_path):
    """Inspect an OpenRouter .eml file to understand its structure."""
    print(f"\n{'='*60}")
    print(f"📧 INSPECTING: {Path(eml_path).name}")
    print(f"{'='*60}")
    
    try:
        with open(eml_path, 'rb') as f:
            msg = email.message_from_bytes(f.read())
        
        # Extract basic email metadata
        subject = msg.get('Subject', 'No Subject')
        sender = msg.get('From', 'Unknown Sender')
        date = msg.get('Date', 'Unknown Date')
        
        print(f"Subject: {subject}")
        print(f"From: {sender}")
        print(f"Date: {date}")
        
        # Extract email body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode('utf-8', errors='ignore')
                elif part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = payload.decode('utf-8', errors='ignore')
                        # Extract text from HTML (simple approach)
                        import re
                        text = re.sub(r'<[^>]+>', '', html_body)
                        body += text
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body += payload.decode('utf-8', errors='ignore')
        
        print(f"\n📄 EMAIL BODY (first 100 lines):")
        print("-" * 60)
        
        lines = body.split('\n')
        for i, line in enumerate(lines[:100]):
            if line.strip():
                print(f"{i+1:3d}: {line}")
        
        if len(lines) > 100:
            print(f"... ({len(lines) - 100} more lines)")
        
        # Look for invoice-specific patterns
        print(f"\n🔍 SEARCHING FOR INVOICE PATTERNS:")
        print("-" * 60)
        
        # Look for receipt number
        receipt_match = re.search(r'#(\d{4}-\d{4})', subject)
        if receipt_match:
            print(f"Receipt Number: {receipt_match.group(1)}")
        
        # Look for date patterns in body
        date_patterns = [
            r'(\w+ \d{1,2}, \d{4})',  # May 17, 2025
            r'(\d{4}-\d{2}-\d{2})',   # 2025-05-17
            r'(\d{1,2}/\d{1,2}/\d{4})', # 5/17/2025
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, body)
            if dates:
                print(f"Found dates: {dates[:5]}")  # Show first 5
                break
        
        # Look for amount patterns
        amount_patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*USD',
            r'Total.*?(\d+\.?\d*)',
            r'Amount.*?(\d+\.?\d*)',
        ]
        
        for pattern in amount_patterns:
            amounts = re.findall(pattern, body)
            if amounts:
                print(f"Found amounts: {amounts[:5]}")  # Show first 5
                break
        
        # Look for OpenRouter specific patterns
        if 'openrouter' in body.lower():
            print("✅ Contains 'OpenRouter' text")
        
        # Look for specific transaction info
        lines = body.split('\n')
        for i, line in enumerate(lines):
            if 'openrouter' in line.lower() and ('$' in line or 'usd' in line.lower()):
                print(f"OpenRouter transaction line {i}: {line.strip()}")
        
    except Exception as e:
        print(f"❌ Error inspecting {eml_path}: {e}")

def main():
    print("🔍 OpenRouter EML Inspector")
    print("=" * 60)
    
    # Find OpenRouter .eml files
    eml_dir = Path('/home/ts/Downloads/coding_llm_provider_invoices')
    openrouter_files = list(eml_dir.glob("Your OpenRouter*.eml"))
    
    if not openrouter_files:
        print("❌ No OpenRouter .eml files found")
        return
    
    print(f"Found {len(openrouter_files)} OpenRouter .eml files")
    
    # Inspect first few files
    for i, eml_file in enumerate(openrouter_files[:3]):
        inspect_openrouter_eml(str(eml_file))
        if i < 2:  # Add separator between files
            print("\n" + "="*60)

if __name__ == "__main__":
    main()

# pdf_extractor.py - PDF Text Extraction Module
# Extracts text content from PDF files for placeholder filling

import logging
from typing import Optional
from dataclasses import dataclass

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result from PDF extraction"""
    success: bool
    text: str
    page_count: int = 0
    error: Optional[str] = None


class PDFExtractor:
    """
    Extracts text content from PDF files.
    Uses pdfplumber for accurate text extraction with layout preservation.
    """
    
    def __init__(self):
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            )
    
    def extract(self, pdf_path: str) -> PDFExtractionResult:
        """
        Extract all text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFExtractionResult with extracted text
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            text_parts = []
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"PDF has {page_count} pages")
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    logger.debug(f"Page {i+1}: extracted {len(page_text or '')} chars")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} total characters from PDF")
            
            return PDFExtractionResult(
                success=True,
                text=full_text,
                page_count=page_count
            )
            
        except FileNotFoundError:
            error = f"PDF file not found: {pdf_path}"
            logger.error(error)
            return PDFExtractionResult(
                success=False,
                text="",
                error=error
            )
        except Exception as e:
            error = f"Error extracting PDF: {str(e)}"
            logger.error(error)
            return PDFExtractionResult(
                success=False,
                text="",
                error=error
            )


def extract_pdf_text(pdf_path: str) -> str:
    """
    Convenience function to extract text from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string (empty if failed)
    """
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path)
    return result.text if result.success else ""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    args = parser.parse_args()
    
    extractor = PDFExtractor()
    result = extractor.extract(args.pdf)
    
    if result.success:
        print(f"\n=== Extracted Text ({result.page_count} pages) ===")
        print(result.text)
    else:
        print(f"Extraction failed: {result.error}")

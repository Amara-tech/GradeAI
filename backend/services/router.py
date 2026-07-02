import os
from typing import Callable, Dict
import pypdf

class DocumentIngestionRouter:
    """
    Handles the initial routing of course materials by inspecting file extensions
    and internal PDF structures to differentiate between digital text, images, 
    and scanned/photographed PDFs.
    """
    
    def __init__(self, text_pdf_handler: Callable[[str], None], 
                 scanned_pdf_handler: Callable[[str], None], 
                 slide_image_handler: Callable[[str], None],
                 docx_handler: Callable[[str], None]):
        """
        Inject the downstream pipeline functions as callbacks.
        """
        self.text_pdf_handler = text_pdf_handler
        self.scanned_pdf_handler = scanned_pdf_handler
        self.slide_image_handler = slide_image_handler
        self.docx_handler = docx_handler
        
        # Supported non-PDF formats
        self.slide_image_extensions = {'.pptx', '.ppt', '.png', '.jpg', '.jpeg', '.tiff', '.docx'}
        self.text_threshold = 20  # Minimum characters to be considered a digital PDF

    def route_document(self, file_path: str) -> None:
        """
        Main entry point to inspect and route the document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source material not found at: {file_path}")
            
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            self._process_pdf_routing(file_path)
        elif ext in self.slide_image_extensions:
            print(f"[Routing] -> Slides/Image Handler for: {file_path}")
            self.slide_image_handler(file_path)
        elif ext == '.docx':
            print(f"[Routing] -> Docx Document Handler for: {file_path}")
            self.docx_handler(file_path)  # Route it here   
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _process_pdf_routing(self, file_path: str) -> None:
        """
        Internal checks to determine if the PDF is digital or a raw image scan.
        """
        # Test 1: Read Sample Bytes for magic number header
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                raise ValueError(f"Corrupted or mislabeled file: {file_path} (Missing %PDF header)")

        try:
            reader = pypdf.PdfReader(file_path)
            
            # Ensure the document isn't empty
            if not reader.pages:
                raise ValueError(f"The PDF file contains no pages: {file_path}")
                
            # Test 2: Page Text Content Check (Looking at Page 1)
            first_page = reader.pages[0]
            extracted_text = first_page.extract_text() or ""
            
            if len(extracted_text.strip()) > self.text_threshold:
                print(f"[Routing] -> Digital PDF Handler for: {file_path}")
                self.text_pdf_handler(file_path)
                return

            # Test 3: Element Check (Looking for renderable fonts or text objects)
            # If there are no fonts defined on the first page, it's structurally an image flat-scan.
            has_fonts = "/Font" in first_page.get("/Resources", {})
            
            if not has_fonts:
                print(f"[Routing] -> Scanned PDF Engine (OCR) for: {file_path}")
                self.scanned_pdf_handler(file_path)
            else:
                # Edge case: Fonts exist but no text was extracted (could be broken encodings or vector paths)
                print(f"[Routing] -> Fallback to Scanned PDF Engine (OCR) for safety: {file_path}")
                self.scanned_pdf_handler(file_path)

        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF structures for {file_path}: {str(e)}")
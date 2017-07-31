from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

fp = open("aaa.pdf", "rb")
parser = PDFParser(fp)
doc = PDFDocument()
parser.set_document(doc)
doc.set_parser(parser)
doc.initialize("")
resource = PDFResourceManager()
device = PDFPageAggregator(resource, laparams=LAParams())
interpreter = PDFPageInterpreter(resource, device)

for page in doc.get_pages():
    interpreter.process_page(page)
    layout = device.get_result()
    for out in layout:
        if hasattr(out, "get_text"):
            print(out.get_text())

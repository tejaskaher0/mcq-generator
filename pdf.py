import pdfkit

# Define path to wkhtmltopdf.exe
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

# Define path to HTML file
path_to_file = 'templates/index.html'

# Point pdfkit configuration to wkhtmltopdf.exe
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

# Convert HTML file to PDF
pdfkit.from_file(path_to_file, output_path='sample1.pdf', configuration=config)

# parsers.py
import werkzeug
from flask_restplus import reqparse

# Requirement for input file and for show on swagger
file_upload = reqparse.RequestParser()
file_upload.add_argument('File',  
                         type=werkzeug.datastructures.FileStorage, 
                         location='files', 
                         required=True, 
                         help='PDF file')

# Requirement for input id and for show on swagger
id_of_file = reqparse.RequestParser()
id_of_file.add_argument('ID',
                         type=int,
                         location='form',
                         required=True,
                         help='ID of file')

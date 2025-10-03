# import sys
# import os

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # import the logger object directly from the local logger.py file
# from logger import logger as logging


# def error_message_detail(error, error_detail: sys):
#     """
#     Generates a detailed error message with file name, line number, and error.
#     """
#     _, _, exc_tb = error_detail.exc_info()
#     file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#     error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
#     return error_message

# class CustomException(Exception):
#     """
#     A custom exception class for handling errors gracefully.
#     """
#     def __init__(self, error_message, error_detail: sys):
#         super().__init__(error_message)
#         self.error_message = error_message_detail(error_message, error_detail=error_detail)

#     def __str__(self):
#         return self.error_message
# DeepLearning_module/notebook/exception.py

import sys
import os
from logger import logger as logging # Imports from the local logger.py

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
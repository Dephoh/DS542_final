import os

def get_file_paths(directory):
   """
   Get paths of all files in a directory and its subdirectories.
   
   Args:
       directory (str): Path to the directory to search
       
   Returns:
       list: List of full file paths
   """
   file_paths = []
   
   # Walk through directory and subdirectories
   for root, dirs, files in os.walk(directory):
       for file in files:
           # Create full file path
           file_path = os.path.join(root, file)
           file_paths.append(file_path)
           
   return file_paths

directory = "./scaled/"
paths = get_file_paths(directory)

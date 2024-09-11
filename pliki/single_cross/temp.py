import os

def rename_files_in_directory(directory, substring):
    for filename in os.listdir(directory):
        if substring in filename:
            new_filename = filename.replace(substring, "")
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")

# Specify the directory and the substring to remove
current_directory = os.getcwd()
substring_to_remove = "_cross"

# Call the function to rename the files
rename_files_in_directory(current_directory, substring_to_remove)

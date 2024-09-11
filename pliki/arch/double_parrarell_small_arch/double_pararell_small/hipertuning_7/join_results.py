def join_text_files(file_list, output_file):
    """
    Joins the contents of multiple text files into one text file.

    :param file_list: List of text file names to be joined
    :param output_file: The name of the output file where the contents will be written
    """
    try:
        with open(output_file, 'w') as outfile:
            for file_name in file_list:
                with open(file_name, 'r') as infile:
                    contents = infile.read()
                    outfile.write(contents)
                    #outfile.write("\n")  # Optional: Add a newline to separate file contents
        print(f"Files have been successfully joined into {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_list = ['hipertuning10001.txt','hipertuning20001.txt','hipertuning30001.txt','hipertuning40001.txt','hipertuning50001.txt','hipertuning99999.txt']
output_file = 'hipertuning_7.txt'
join_text_files(file_list, output_file)

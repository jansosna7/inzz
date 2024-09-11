def join_text_files(output_file):
    """
    Joins the contents of multiple text files into one text file.

    :param file_list: List of text file names to be joined
    :param output_file: The name of the output file where the contents will be written
    """
    prefix = "hipertuning100"
    suffix = ".txt"
    try:
        with open(output_file, 'w') as outfile:
            for file_nr in range(50,70):
                file_name = prefix + str(file_nr)+suffix
                with open(file_name, 'r') as infile:
                    contents = infile.read()
                    outfile.write(contents)
        print(f"Files have been successfully joined into {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
output_file = 'hipertuning_2.txt'
join_text_files(output_file)

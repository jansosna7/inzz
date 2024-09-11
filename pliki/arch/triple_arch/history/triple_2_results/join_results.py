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
        print(f"Files have been successfully joined into {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_list = ['hipertuning20000_0.txt',
             'hipertuning20000_1.txt',
             'hipertuning20000_2.txt',
             'hipertuning20000_3.txt',
             'hipertuning20000_4.txt',
             'hipertuning20001_0.txt',
             'hipertuning20001_1.txt',
             'hipertuning20001_2.txt',
             'hipertuning20001_3.txt',
             'hipertuning20001_4.txt',
             'hipertuning20001_5.txt',
             'hipertuning20001_6.txt',
             'hipertuning20001_7.txt',
             'hipertuning20001_8.txt',
             'hipertuning20001_10.txt',
             'hipertuning20001_13.txt',
             'hipertuning20001_14.txt',
             'hipertuning20001_15.txt',
             'hipertuning20001_16.txt',
             'hipertuning20001_17.txt']

output_file = 'hipertuning_2.txt'
join_text_files(file_list, output_file)

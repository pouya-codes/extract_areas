import pandas as pd

class ReadMetadata:
    def __init__(self, file_path, sheet_name=None):
        """
        Initialize the ReadMetadata class with the path to the Excel file and optional sheet name.
        
        :param file_path: Path to the Excel file containing metadata.
        :param sheet_name: Name of the sheet to read from the Excel file. If None, the first sheet is used.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.metadata = self._read_excel()

    def _read_excel(self):
        """
        Read the Excel file and store the metadata in a dictionary.
        
        :return: Dictionary with slide names as keys and corresponding core labels as values.
        """
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        metadata = {}
        
        # Assuming the first row contains the slide names and the first column contains the core numbers
        slide_names = df.columns[1:-2]
        for slide in slide_names:
            slide = str(slide)  # Ensure slide name is a string
            metadata[slide] = {}
            for index, row in df.iterrows():
                core_number = str(row.iloc[0])  # Ensure core number is a string
                if slide in row:
                    metadata[slide][core_number] = row[slide]
                else:
                    metadata[slide][core_number] = row[int(slide)]
        print(metadata)
        return metadata
    

    def get_metadata(self, slide_name, core_number):
        """
        Retrieve metadata for a given slide name and core number.
        
        :param slide_name: The slide name for which metadata is to be retrieved.
        :param core_number: The core number for which metadata is to be retrieved.
        :return: Metadata for the specified slide name and core number.
        """
        return self.metadata.get(slide_name, {}).get(core_number, None)
    
    def get_number_of_cores(self):
        """
        Get the number of cores for which metadata is available.
        
        :return: Number of cores.
        """
        return len(next(iter(self.metadata.values())))

# Example usage:
metadata_reader = ReadMetadata('CPQA.xlsx', 'BRAFV600E assessments')
slide_metadata = metadata_reader.get_metadata('240', '1')
print(metadata_reader.get_number_of_cores())    
print(slide_metadata)